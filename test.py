import argparse, os, logging
import torch
from train import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test.run")

parser = argparse.ArgumentParser()
# basic config
parser.add_argument('--random_seed', type = int, default=2026, help='Random Seed num')

# data loader
parser.add_argument('--data', type = str, default='all', help = 'dataset type')
parser.add_argument('--input_dim', type=int, default=4, help='input dimension')
parser.add_argument('--d_input', type=int, default=1, help='input dimension')
parser.add_argument('--n_return', type=int, default=78, help='number of return series')

parser.add_argument('--model_type', type = str, default='DeepHAR', help='model type')
parser.add_argument('--model_name', type = str, default='model', help='Directory containing params.json')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')

# forecasting task
parser.add_argument('--seq_len', type=int, default=22, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

# model define
parser.add_argument('--lstm_layers', type=int, default=2, help='RNN hidden layers') 
parser.add_argument('--lstm_hidden_dim', type=int, default=64, help='RNN hidden dim')
parser.add_argument('--n_heads', type=int, default=8, help='number of heads(MHA)')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

# optimization
parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='QLIKE', help='loss function')
parser.add_argument('--init_tries', type=int, default=20, 
                    help='maximum number of re-initialization attempts')
parser.add_argument('--clamp_min', type=float, default=0.01, 
                    help='minimum threshold for model outputs (to prevent dead neurons)')

if __name__ == "__main__":
    configs = parser.parse_args()
    set_seed(configs.random_seed)

    # best hparams
    # configs.learning_rate = 0.0001
    # configs.n_heads = 8
    # configs.dropout = 0.1
    # configs.lstm_hidden_dim = 64
    # configs.lstm_layers = 2

    configs.data_path = os.path.join(configs.root_path, configs.data)

    configs.setting = 'rs{}_dr{}_lr{}_ll{}_lh{}_nh{}'.format(
        configs.random_seed, configs.dropout, configs.learning_rate,
        configs.lstm_layers, configs.lstm_hidden_dim, configs.n_heads
    )

    configs.model_dir = os.path.join(configs.model_name, configs.model_type, configs.data) 
    ckpt_path = os.path.join(configs.model_dir, configs.setting, "checkpoint.pth")   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs.device = device

    dataset_all = Dataset_all(configs)
    test_set, test_loader = dataset_all._get_data("test")

    model = UnifiedModel(configs).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    summary_test = evaluate(model, test_set, test_loader, configs,
                            loss_type=["QLIKE", "MSE"])
    logger.info(summary_test)

    # Save Result
    result_dir = os.path.join("result", configs.model_type, configs.data) 
    os.makedirs(result_dir, exist_ok=True)  

    out_path = os.path.join(result_dir, f"{configs.setting}.json")
    with open(out_path, "w") as f:
        json.dump(summary_test, f, indent=2)

    logger.info(f"Saved test results to: {out_path}")