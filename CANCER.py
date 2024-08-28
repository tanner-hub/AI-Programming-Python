from sklearn .datasets import load_breast_cancer

cancer = load_breast_cancer()
print("___Feature_Names___\n", cancer['feature_names'], "\n")
print("___Data___\n", cancer['data'], "\n")
print("___Target_Names___\n", cancer['target_names'], "\n")