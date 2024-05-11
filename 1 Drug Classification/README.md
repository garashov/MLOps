### Project Description
In this guide, we will walk you through the process of setting up accounts and environments, creating a CI/CD pipeline, and optimizing the entire process.

We will be using scikit-learn pipelines to train our random forest algorithm and build a drug classifier. After training, we will automate the evaluation process using CML. Finally, we will build and deploy the web application to Hugging Face Hub.

From training to evaluation, the entire process will be automated using GitHub actions. All you have to do is push the code to your GitHub repository, and within two minutes, the model will be updated on Hugging Face with the updated app, model, and results.


### Techonological Stack
- Python
- Hugging Face
- GitHub Actions
- CML - Continuous Machine Learning (CML) is an open-source library that allows you to implement continuous integration within your machine learning projects.
- Makefile - A Makefile is a file that consists of a set of instructions used by make command to automate various tasks, such as compiling code, running tests, setting up environments, preprocessing data, training and evaluating models, and deploying models.


### References
- https://www.datacamp.com/tutorial/ci-cd-for-machine-learning
- [Dataset](https://www.kaggle.com/datasets/prathamtripathi/drug-classification)