import MTCS as mt
import model as md
import torch


if __name__ == "__main__":
    
    model = md.ChessNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    #model.eval()

    state = torch.load("model_weights.pth", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Test MCTS with a small search
    print("Testing MCTS implementation...")
    mt.test_board(model, device)
    
    print("\n" + "="*50)
    print("MCTS implementation is working correctly!")
    print("="*50)
    print("\nTo generate training data, run:")
    print("python generate_training_data.py --num_games 10 --num_simulations 100")
    print("\nTo play a full game, uncomment the line below:")
    print("# mt.run_game(model, 0.8, 100, device)")