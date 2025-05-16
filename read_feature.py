import os 
import pickle
def main():
    example = "sample_data/output_rgb_segmentations.pkl"
    with open(example, "rb") as f:
        data = pickle.load(f)

    print("Data structure:")
    print(data.keys())
    print(data)
    
if __name__ == "__main__":
    main()