import os 
import pickle
def main():
    exmaple = "sample_data/output_rgb_detections.pkl"
    with open(exmaple, "rb") as f:
        data = pickle.load(f)
    print(data)

if __name__ == "__main__":
    main()