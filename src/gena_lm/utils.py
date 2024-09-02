import copy, string
import numpy as np

tab = str.maketrans("ACTGN", "TGACN")

def reverse_complement(seq):
    return seq.translate(tab)[::-1]


def get_service_token_encodings(tokenizer):
    SEP_token_id = tokenizer.sep_token_id
    CLS_token_id = tokenizer.cls_token_id

    CLS_encoding = {
        "input_ids": np.array(CLS_token_id).reshape(-1, 1),
        "token_type_ids": np.array([0]).reshape(-1, 1),
        "attention_mask": np.array([1]).reshape(-1, 1),
    }

    SEP_encoding = {
        "input_ids": np.array(SEP_token_id).reshape(-1, 1),
        "token_type_ids": np.array([0]).reshape(-1, 1),
        "attention_mask": np.array([1]).reshape(-1, 1),
    }

    return {"CLS": CLS_encoding, "SEP": SEP_encoding}


def concatenate_encodings(token_dicts):
    """
    encodings are dictionaries containing arrays, typically obtained calling
    tokenizer(sequence,
                            max_length = max_seq_len,
                            padding=False,
                            truncation=False,
                            add_special_tokens=False,
                            return_tensors="np")
    Given several input dicts, this function will concatenate matching arrays
    Note that all dict should have the same keys
    """

    assert len(set([len(d.keys()) for d in token_dicts])) == 1, "Provided encoding dicts have different number of keys"
    for i in token_dicts[0].keys():
        for j in token_dicts:
            assert i in j.keys(), "Key " + i + " was not found in dict keys:\n" + str(j.keys())

    key = list(token_dicts[0].keys())[0]
    return {key: np.concatenate([d[key][0] for d in token_dicts]).reshape(1, -1) for key in token_dicts[0].keys()}


# def symmetric_truncate(
#     mid_encoding, n_service_tokens, max_seq_len
# ):
#     """
#     Truncate sequence encodding symmetrically, i.e. removing same number of tokens from left and right
#     """
#     final_length = max_seq_len-n_service_tokens
#     original_len = len(list(mid_encoding.values())[0])
#     assert original_len>=final_length, f"Unable to truncate sequence with length f{original_len} to f{max_seq_len-n_service_tokens}"
#     if original_len==final_length:
#         return mid_encoding

#     trim_length = original_len-final_length
#     trim_left = trim_length//2
#     trim_right = trim_length - trim_left
#     truncated = {}
#     for key in mid_encoding.keys():
#         truncated[key] = mid_encoding[key][trim_left:original_len-trim_right]
#     assert len(list(truncated.values()[0]))==final_length
#     return truncated


def symmetric_pad_and_truncate_context(
    left_encoding, right_encoding, mid_encoding, n_service_tokens, max_seq_len, PAD_id
):
    """
    Given encodings of context (left_encoding and right_encoding)
    and encoding of the "target" middle DNA fragment
    perform truncation or padding with PAD_id tokens so that
    1) returned encoding has length max_seq_len-n_service_tokens
    2) padding, if needed, will be added to the right-most part of the
    sequence
    3) truncation, if needed, will be symmetric, i.e. will be done so that
    left and right context lengths are as close to each other as possible
    4) left context part will be truncated from the left, right context part
    will be truncated from the right side.

    encodings are dictionaries containing arrays, typically obtained calling
    tokenizer(sequence,
                            max_length = max_seq_len,
                            padding=False,
                            truncation=False,
                            add_special_tokens=False,
                            return_tensors="np")

    if left/right/mid encoding is None, it will be initiallized as an empty array

    Returns tuple: (left_encoding, right_encoding, mid_encoding, padding_encoding)
    """

    def safe_return(left_encoding, mid_encoding, right_encoding, padding):
        # check sum of left_encoding, right_encoding, and mid_encoding length
        assert (
            len(mid_encoding["input_ids"][0])
            + len(left_encoding["input_ids"][0])
            + len(right_encoding["input_ids"][0])
            + len(padding["input_ids"][0])
            + n_service_tokens
            == max_seq_len
        )

        # check that all arrays of one encoding have the same length
        for e in left_encoding, mid_encoding, right_encoding, padding:
            assert len(set([len(x[0]) for x in e.values()])) == 1

        return left_encoding, right_encoding, mid_encoding, padding

    assert mid_encoding["input_ids"].shape[1] > 0, """Mid part of encoding has 0 length. 
    This may happen if input sequence was an empty string."""

    # initialize default empty padding (=no padding)
    empty_array = {
        "input_ids": np.array([], dtype=np.int32).reshape(1, -1),
        "token_type_ids": np.array([], dtype=np.int32).reshape(1, -1),
        "attention_mask": np.array([], dtype=np.int32).reshape(1, -1),
    }

    for encoding in [mid_encoding, left_encoding, right_encoding]:
        if encoding is not None:
            assert np.all(encoding["attention_mask"][0] == 1)
            assert np.all(encoding["token_type_ids"][0] == 0)

    if left_encoding is None:
        left_encoding = copy.deepcopy(empty_array)
    if right_encoding is None:
        right_encoding = copy.deepcopy(empty_array)

    L_mid = len(mid_encoding["input_ids"][0])
    L_left = len(left_encoding["input_ids"][0])
    L_right = len(right_encoding["input_ids"][0])

    # case I. mid encoding >= max_seq_len; don't add context & trim target if needed
    if L_mid + n_service_tokens >= max_seq_len:
        trim_length = L_mid + n_service_tokens - max_seq_len
        assert trim_length >= 0
        if trim_length == 0:
            return safe_return(empty_array, mid_encoding, empty_array, empty_array)
        trim_left = trim_length // 2
        trim_right = trim_length - trim_left
        truncated = {}

        for key in mid_encoding.keys():
            truncated[key] = mid_encoding[key][0][trim_left : L_mid - trim_right].reshape(1, -1)

        return safe_return(empty_array, truncated, empty_array, empty_array)

    # case II. target+context encoding < max_seq_len, we need to pad
    elif L_mid + L_left + L_right + n_service_tokens <= max_seq_len:
        n_pads = max_seq_len - (L_mid + L_left + L_right + n_service_tokens)
        if n_pads == 0:
            padding = copy.deepcopy(empty_array)
            return safe_return(left_encoding, mid_encoding, right_encoding, padding)
        assert n_pads > 0

        padding = {
            "input_ids": np.array([PAD_id] * n_pads, dtype=np.int32).reshape(1, -1),
            "token_type_ids": np.array([mid_encoding["token_type_ids"][0][0]] * n_pads).reshape(1, -1),
            "attention_mask": np.array([0] * n_pads, dtype=np.int32).reshape(1, -1),
        }

        # right_encoding["input_ids"] = np.concatenate(
        #     [
        #         right_encoding["input_ids"][0],
        #         [PAD_id] * n_pads,
        #     ]
        # ).reshape(1, -1)

        # right_encoding["token_type_ids"] = np.concatenate(
        #     [
        #         right_encoding["token_type_ids"][0],
        #         [right_encoding["token_type_ids"][0][0]] * n_pads,
        #     ]
        # ).reshape(1, -1)

        # right_encoding["attention_mask"] = np.concatenate(
        #     [
        #         right_encoding["attention_mask"][0],
        #         [0] * n_pads,
        #     ]
        # ).reshape(1, -1)

        return safe_return(left_encoding, mid_encoding, right_encoding, padding)
    # case III. target+context encoding > max_seq_len, we need to trim
    elif L_mid + L_left + L_right + n_service_tokens > max_seq_len:
        # compute trimming. The aims are to
        # a) make the total length == max_seq_len
        # b) make the left and right context size as close to each other as possible
        oversize = L_mid + L_left + L_right + n_service_tokens - max_seq_len
        if L_left >= L_right:
            trim_left = oversize / 2.0 + min((L_left - L_right) / 2.0, oversize / 2.0)
            trim_right = max(0, (oversize - (L_left - L_right)) / 2.0)
        else:
            trim_right = oversize / 2.0 + min((L_right - L_left) / 2.0, oversize / 2.0)
            trim_left = max(0, (oversize - (L_right - L_left)) / 2.0)
        assert (int(trim_right) == trim_right) == (int(trim_left) == trim_left)
        if int(trim_right) != trim_right:
            trim_left += 0.5
            trim_right -= 0.5
        assert (int(trim_right) - trim_right) == (int(trim_left) - trim_left) == 0
        assert oversize == trim_left + trim_right

        trim_left = int(trim_left)
        trim_right = int(trim_right)

        assert trim_left >= 0
        assert trim_left < L_left
        assert trim_right < L_right

        trim_right = L_right - trim_right

        for key in ["input_ids", "token_type_ids", "attention_mask"]:
            left_encoding[key] = left_encoding[key][0][trim_left:].reshape(1, -1)
            right_encoding[key] = right_encoding[key][0][:trim_right].reshape(1, -1)

        padding = copy.deepcopy(empty_array)
        return safe_return(left_encoding, mid_encoding, right_encoding, padding)
    else:
        raise ValueError("Unexpected encoding length")
