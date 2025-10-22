import torch


class DummyMetrics:
    def __call__(self, prediction):

        print("#"*80)
        print(predictions.)
        for i, p in enumerate(prediction.predictions):
            print(f"Prediction #{i}:", p)
            print("="*80, flush = True)
        print("#"*80)