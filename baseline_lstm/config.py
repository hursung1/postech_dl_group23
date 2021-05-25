import argparse

def get_params():
    parser = argparse.ArgumentParser(description="Sentiment Classification")
    parser.add_argument("--is_train", type=bool, )
    parser.add_argument("--emb_type", type=str, default="GloVe")
    parser.add_argument("--emb_data", type=str, default="6B")
    parser.add_argument("--emb_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    
    params = parser.parse_args()
    return params