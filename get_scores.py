
import argparse
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,accuracy_score

from utils import *

def main(args):

	assert os.path.isfile(args.filepath),'File path not found... must provide a valid file path...'

	print('Preprocessing data...')

	processed_df = prepare_data(args.filepath, args.date_col, args.id_col, args.price_col)

	print('Splitting data into training and test sets...')
	
	X_train, X_test, y_train, y_test = split_data(processed_df, args.date_col, args.id_col, args.price_col)

	print('Training set size : ', len(X_train))
	print('Test set size : ', len(X_test))

	y_actual = y_test

	model = LinearRegression()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	
	print('Mean Absolute Error: ', mean_absolute_error(y_actual,y_pred))
	print('Mean Squared Error:', np.sqrt(mean_squared_error(y_actual,y_pred)))
	print('R2 Score: ', r2_score(y_actual,y_pred))

	y_pred = min_max_scaler(pd.DataFrame(y_pred))

	ids = pd.DataFrame(X_test.index)
	scores = pd.concat([ids, y_pred], axis=1, join_axes=[y_pred.index])
	scores.columns = ['id','score']
	with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
		print(scores)
	
	scores.to_csv('score_report.csv')
	print('Score report is saved to local disk.')

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-f', '--filepath',
		help="Dataset file path to use as input; example: your_file_path.csv",
		required=True, type=str)
	parser.add_argument(
		'-d', '--date_col',
		help="Date column name in dataset file",
		required=True, type=str)
	parser.add_argument(
		'-id', '--id_col',
		help="Customer ID column name in dataset file",
		required=True, type=str)
	parser.add_argument(
		'-p', '--price_col',
		help="Price column name in dataset file",
		required=True, type=str)

	args = parser.parse_args()

	if len(args) != 4:
		print("This application expects 4 arguments")

	main(args)


	