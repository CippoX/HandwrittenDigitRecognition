# HANDWRITTEN DIGIT RECOGNITION USING SVM, RANDOM FORESTS, NAIVE BAYES CLASSIFIER, AND K-NN

**Ca’ Foscari University of Venice**  
Master’s Degree in Computer Science and Information Technology  

[LM-18]  
Palmisano Tommaso, 886825  


## I. Abstract
Handwritten digit recognition might seem trivial due to modern advancements in AI, but solving it with classical programming methods proves to be quite challenging. This project explores the use of Machine Learning techniques to address this problem, analyzing both Discriminative and Generative classifiers. The models evaluated include Support Vector Machines (SVM), Random Forests, Naive Bayes, and k-Nearest Neighbors (k-NN).

## II. Introduction
Handwritten digit recognition is a fundamental problem in machine learning, often solved using the MNIST dataset. Two main approaches exist: Discriminative classifiers, which model the decision boundary between classes (e.g., SVM and Random Forests), and Generative classifiers, which model the joint probability distribution of data and labels (e.g., Naive Bayes). k-NN, a non-parametric model, is also evaluated for its simplicity and effectiveness.

## III. Machine Learning Models
### **1. Support Vector Machines (SVM)**
SVMs are based on finding an optimal hyperplane that maximizes the margin between two classes. Different kernel functions are used to handle non-linearly separable data, including:
- **Linear Kernel**: Works well for linearly separable data.
- **Polynomial Kernel**: Maps data to a higher dimension using a polynomial function.
- **Radial Basis Function (RBF) Kernel**: Maps data to an infinite-dimensional space, making it highly effective for complex distributions.

Grid search and cross-validation were used to optimize hyperparameters, improving accuracy and generalization. The best model achieved **98.4% accuracy** with an **RBF kernel**.

### **2. Random Forests**
Random Forests combine multiple Decision Trees to improve classification accuracy. The key features of this approach include:
- **Bootstrap Aggregation (Bagging)**: Uses random subsets of the data to train individual trees.
- **Random Feature Selection**: Selects a subset of features at each split to reduce overfitting.
- **Majority Voting**: Combines predictions from multiple trees for improved accuracy.

The best Random Forest model achieved **97.8% accuracy** with **750 estimators** and a maximum depth of **40**.

### **3. Naive Bayes Classifier**
Naive Bayes is a probabilistic classifier based on Bayes’ theorem, assuming feature independence. This model:
- Uses prior probabilities of each digit class.
- Models each pixel distribution using a **Beta distribution**.
- Calculates the posterior probability to classify new digits.

Although highly efficient (**2.66s training time**), Naive Bayes had lower accuracy (**72.3%**), largely due to the strong independence assumption.

### **4. k-Nearest Neighbors (k-NN)**
k-NN is a simple instance-based learning algorithm where classification is based on the majority vote of the **k** closest neighbors. The main characteristics include:
- **Memory-intensive**, as all training data must be stored.
- **Computation-heavy**, as distances must be computed for each new query.
- **Highly accurate**, with the best model achieving **97.3% accuracy** for **k=3**.

## IV. Execution Time and Performance
- **SVM (RBF Kernel)**: **6.1 min** training time, highest accuracy (**98.4%**).
- **Random Forests**: **3.4 min** training time, competitive accuracy (**97.8%**).
- **Naive Bayes**: **2.66s** training time, but lowest accuracy (**72.3%**).
- **k-NN**: **271s evaluation time**, due to high memory and computation costs.

## V. Conclusions
This study compared four machine learning models for handwritten digit recognition. **SVM with RBF kernel** provided the best accuracy, but at a high computational cost. **Random Forests** offered a strong trade-off between performance and efficiency. **Naive Bayes**, while computationally efficient, struggled with accuracy due to feature independence assumptions. **k-NN** performed well but suffered from scalability issues.

In conclusion, the choice of the best model depends on the application: SVM for accuracy, Random Forests for balanced performance, Naive Bayes for speed, and k-NN for simplicity in small datasets.

## References
1. Yann LeCun, Corinna Cortes, and Christopher J.C. Burges. *The MNIST Database of Handwritten Digits.*  
2. Joshua Starmer. *StatQuest: Support Vector Machines & Random Forests.*  
3. Andrea Torsello. *Machine Learning and AI Course Materials, 2024.*  
4. Marcello Pelillo. *Statistical Learning Theory, 2024.*

