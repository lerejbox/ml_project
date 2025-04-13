from Q1_functions import predict_tags, estimate_emission_parameters

def main():
    # 1. Learn emission parameters from the training file
    emission_params = estimate_emission_parameters('./EN/train')
    
    # 2. Apply the system to the development set
    predict_tags(emission_params, './EN/dev.in', './EN/dev.p1.out')
    
    # 3. Calculate precision, recall, and F-score using the EvalScript/evalResult.py

if __name__ == "__main__":
    main()
