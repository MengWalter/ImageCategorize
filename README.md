# ImageCategorize
# categorize the test images into 18 categories
The training set are a fold contains 18 sub folders, each of which has one category of images

After pretrain the image, I convert each image into a 1*n matrix and concat them, then export the matrix into .npy files, thus we could have 18 .npy files for each category (Use 500 of each category image as the validate data set)
Label each category when import them in basic_model.py
Training the model with keras sequential()
Make prediction in test dataset
