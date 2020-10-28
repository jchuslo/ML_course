'''Functions which I created for the 2nd homework assignment in COMP 135'''


############################################################




def test_polynomials(polynomials=list()):
    '''Generates a series of polynomial regression models on input data.
       Each model is fit to the data, then used to predict values of that
       input data.  Predictions and mean squared error are collected and
       returned as two lists.
    
    Args
    ----
    polynomials : list of positive integer values
        Each value is the degree of a polynomial regression model, to be built.
    
    Returns
    -------
    prediction_list: list of arrays ((# polynomial models) x (# input data))
        Each array contains the predicted y-values for input data.
    error_list: list of error values ((# polynomial models) x 1)
        Each value is the mean squared error (MSE) of the model with 
        the associated polynomial degree.
    '''
    
    predictions = []
    errors = []
    polynomials = sorted(polynomials)
    for deg in polynomials:
        if deg==1:
            linearRegression = LinearRegression()
            linearRegression.fit(x,y)
            yPredict_lin = linearRegression.predict(x)
            mse = mean_squared_error(y, yPredict_lin)
            
            predictions.append(yPredict_lin)
            errors.append(mse)
        
        elif deg>1:
            poly = PolynomialFeatures(degree=deg)
            x_poly = poly.fit_transform(x)
            linearRegression = LinearRegression()
            linearRegression.fit(x_poly, y)
            yPredict = linearRegression.predict(x_poly)
            mse = mean_squared_error(y, yPredict)
            
            predictions.append(yPredict)
            errors.append(mse)
            
            del poly
            
        else:
            print("Please select a polynomial degree >= 1")
            raise ValueError


    prediction_list = list(predictions)
    error_list = list(errors)
        
    return prediction_list, error_list





########################################






'''Function to create k folds of data for cross validation'''


    def make_folds(num_folds=1):
'''Splits data into num_folds separate folds for cross-validation.
   Each fold should consist of M consecutive items from the
   original data; each fold should be the same size (we will assume 
   that  the data divides evenly by num_folds).  Every data item should 
   appear in exactly one fold.
   
   Args
   ----
   num_folds : some positive integer value
       Number of folds to divide data into.
       
    Returns
    -------
    x_folds : list of sub-sequences of original x-data 
        There will be num_folds such sequences; each will 
        consist of 1/num_folds of the original data, in
        the original order.
    y_folds : list of sub-sequences of original y data
        There will be num_folds such sequences; each will 
        consist of 1/num_folds of the original data, in
        the original order.
   '''

x_folds=[]
y_folds=[]
x_parts = np.split(x,num_folds).copy()
y_parts = np.split(y,num_folds).copy()

inx = [x for x in range(num_folds)]
random.shuffle(inx)

for i in range(num_folds):
    x_folds.append(x_parts[inx[i]].copy())
    y_folds.append(y_parts[inx[i]].copy())

x_folds = list(x_folds)
y_folds = list(y_folds)

return x_folds, 




########################################





'''Cross validation function'''

degrees = [1,2,3,4,5,6,10,11,12]
x_folds, y_folds = make_folds(5)
error_matrix = np.zeros([len(degrees),3])

for j in degrees:

    error_train=[]
    error_test=[]

    for i in range(5):
        x_copy, y_copy = x_folds.copy(), y_folds.copy()
        x_test, y_test = x_copy.pop(i), y_copy.pop(i)
        x_train, y_train = x_copy, y_copy
        x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)

        poly = PolynomialFeatures(degree=j)
        
        x_poly_train = poly.fit_transform(x_train)
        linearRegression_train = LinearRegression()
        
        linearRegression_train.fit(x_poly_train, y_train)
        yPredict_train = linearRegression_train.predict(x_poly_train)
        mse = mean_squared_error(y_train, yPredict_train)
        error_train.append(mse)

        x_poly_test = poly.fit_transform(x_test)
        yPredict_test = linearRegression_train.predict(x_poly_test)
        mse = mean_squared_error(y_test, yPredict_test)
        error_test.append(mse)


    avg_mse_train = sum(error_train) / len(error_train)
    avg_mse_test = sum(error_test) / len(error_test)

    error_matrix[degrees.index(j),0] = j
    error_matrix[degrees.index(j),1] = avg_mse_train
    error_matrix[degrees.index(j),2] = avg_mse_test

'''Plot'''
plt.plot(error_matrix[:,0],error_matrix[:,1], color='blue', linewidth=1, linestyle='solid', label="mean training MSE")
plt.plot(error_matrix[:,0],error_matrix[:,2], color='red', linewidth=2, linestyle='dashed', label="mean testing MSE")
plt.title('Average MSE for training and testing data, POLYNOMIAL Models')
plt.xticks(np.arange(1, max(degrees)+1, step=1))
plt.xlabel('Polynomial Degree (of model)')
plt.ylabel('Average MSE')
plt.legend()
plt.show()

'''Table'''
table = pd.DataFrame({'Polynomial Degree': error_matrix[:, 0], 'Average training MSE': error_matrix[:, 1], 'Average testing MSE': error_matrix[:, 2]})
table





###################################







lambdas = list(np.logspace(-2, 2, base=10, num=50))
# Ridge regression regularization strengths


# Polynomial model of degree 3 has the minimum testing error
# using cross validation.
deg = 3


x_folds, y_folds = make_folds(5)
error_matrix = np.zeros([len(lambdas),3])

'''Cross validation'''
for r in lambdas:
    
    error_train=[]
    error_test=[]

    for i in range(5):
        x_copy, y_copy = x_folds.copy(), y_folds.copy()
        x_test, y_test = x_copy.pop(i), y_copy.pop(i)
        x_train, y_train = x_copy, y_copy
        x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)

        poly_ridge = make_pipeline(PolynomialFeatures(deg), Ridge(alpha=r))
        poly_ridge.fit(x_train, y_train)
        yPr_train = poly_ridge.predict(x_train)

        mse = mean_squared_error(y_train, yPr_train)
        error_train.append(mse)

        yPr_test = poly_ridge.predict(x_test)
        mse = mean_squared_error(y_test, yPr_test)
        error_test.append(mse)


    avg_mse_train = sum(error_train) / len(error_train)
    avg_mse_test = sum(error_test) / len(error_test)

    error_matrix[lambdas.index(r),0] = r
    error_matrix[lambdas.index(r),1] = avg_mse_train
    error_matrix[lambdas.index(r),2] = avg_mse_test
    

'''Plot'''    
plt.plot(error_matrix[:,0],error_matrix[:,1], color='blue', linewidth=1, linestyle='solid', label="average training MSE, Ridge")
plt.plot(error_matrix[:,0],error_matrix[:,2], color='red', linewidth=2, linestyle='dashed', label="average testing MSE, Ridge")
plt.title('Average MSE for training and testing data, RIDGE Regression')
matplotlib.pyplot.xscale('log')
plt.xlabel('Ridge Regularization Penalty Value')
plt.ylabel('Average MSE')
plt.legend(loc='center left')
plt.show()

'''Table'''
table = pd.DataFrame({'Ridge Regularization Penalty Value': error_matrix[:, 0], 'Average training MSE': error_matrix[:, 1], 'Average testing MSE': error_matrix[:, 2]})
table
    