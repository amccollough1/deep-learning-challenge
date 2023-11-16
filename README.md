# deep-learning-challenge

# Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years.


# Instructions
Step 1: Preprocess the Data

Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

1.Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:

  What variable(s) are the target(s) for your model?
  
  IS_SUCCESSFULL

  What variable(s) are the feature(s) for your model?
 
  Everything else with the exception of the Name and EIN due to being dropped early in the code. 

2.Drop the EIN and NAME columns.

3.Determine the number of unique values for each column.

4.For columns that have more than 10 unique values, determine the number of data points for each unique value.

5.Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.

6.Use pd.get_dummies() to encode categorical variables.

7.Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

8.Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

Step 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1.Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

2.Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

3.Create the first hidden layer and choose an appropriate activation function.

4.If necessary, add a second hidden layer with an appropriate activation function.

5.Create an output layer with an appropriate activation function.

6.Check the structure of the model.

7.Compile and train the model.

8.Create a callback that saves the model's weights every five epochs.

9.Evaluate the model using the test data to determine the loss and accuracy.

1.Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

Step 3: Optimize the Model

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:

Dropping more or fewer columns.
Creating more bins for rare occurrences in columns.
Increasing or decreasing the number of values for each bin.
Add more neurons to a hidden layer.
Add more hidden layers.
Use different activation functions for the hidden layers.
Add or reduce the number of epochs to the training regimen.

Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

1.Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.

2.Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.

3.Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

4.Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

5.Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

Step 4: Write a Report on the Neural Network Model
------------------------------------------------------
The purpose of this analysis is to aid a nonprofit foundation, namely Alphabet Soup , build a model to help select future applicants for funding with the best chance of success in their ventures. 

Starting with data preprossing, it is important to identify the target and feature variables of the model. In this model, "IS_SUCCESSFUL" is our target variable and everything else, with the exception of EIN are our feature variables. 

![image](https://github.com/amccollough1/deep-learning-challenge/assets/133404805/586bac04-b4da-40af-8b40-52b0f772688f)


In the optimization, I selected 4 neuron,  3 layers, 3 activation functions to optimze my model performance. This model successfully acheived a 75% accuracy score or higher. Which allows it to be a useful model. To optimize the model's performance, I kept thee "NAME" feature in the data set, then proceeded with binning. Then I added a third hidden layer and adjusted the number of layers of the others. 
