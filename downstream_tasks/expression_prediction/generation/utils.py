def pad_to_length(start, end, strand, required_length, chrom):
    length = end - start 
    diff = required_length - length
    if diff < 0: # length is greaterr
        if strand == "+":
            start -= diff 
        elif strand == '-':
            end -= diff
        else:
            raise Exception('Wrong strand')
    elif diff > 0:
        lpad, rest = divmod(diff, 2)
        rpad = lpad + rest
        if strand == "+":
            start -= lpad
            end += rpad
        elif strand == '-':
            start -= rpad
            end += lpad
        else:
            raise Exception('Wrong strand')
    if start < 0 or end >= len(chrom):
        raise NotImplementedError()
    return start, end

def cut_seq(genome, chrom, start, end, strand):
    if strand == '+':
        return genome[chrom][start:end].upper()
    elif strand == "-":
        return genome[chrom][start:end].reverse_complement().upper()

def calc_ident(seq1, seq2):
    assert len(seq1) == len(seq2)
    cm = 0
    for c1, c2 in zip(seq1, seq2):
        if c1 == c2:
            cm += 1
    return cm / len(seq1)