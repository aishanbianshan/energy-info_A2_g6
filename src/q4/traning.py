#!/usr/bin/env python3

import math

# global variables
x_1=0.04
x_2= 0.2
y_1 =0.5
alpha = 0.4
threshold =0.0001



def sigmoid(x):
	return 1/(1+math.e ** -x)

def sigmoid_derivative(a):
	return a*(1-a)

def Error_total(y_out, y_1):
	return(y_out - y_1)**2

def RMSE_metric(MSE):
	return math.sqrt(MSE)

def E_d_total(y_out, y_1):
	return 2*(y_out-y_1)

def forward_and_back_propagation(w_1,w_2,w_3,w_4,w_5,w_6):
	# forward propagation
	sum_h1 = w_1 * x_1 + w_2 * x_2 
	out_h1 = sigmoid(sum_h1)
	sum_h2 = w_3 * x_1 + w_4 * x_2
	out_h2 = sigmoid(sum_h2) 
	sum_y1 = w_5 * out_h1 + w_6 * out_h2
	out_y1 = sigmoid(sum_y1)
	print("sum_h1=",sum_h1)
	print("out_h1=",out_h1)
	print("sum_h2=",sum_h2)
	print("out_h2=",out_h2)
	print("sum_y1=",sum_y1)
	print("out_y1=",out_y1)
	MSE = Error_total(out_y1,y_1)
	RMSE = RMSE_metric(MSE)
	print("E_total =",MSE)
	print("RMSE=",RMSE)
	print("E'_total =",E_d_total(out_y1,y_1))
	print()
 	# back_propagation
	pd_w6 = 2*(out_y1 -y_1)*sigmoid_derivative(out_y1)*out_h2
	pd_w5 = 2*(out_y1 -y_1)*sigmoid_derivative(out_y1)*out_h1
	pd_w4 = 2*(out_y1 -y_1)*sigmoid_derivative(out_y1)*w_6*sigmoid_derivative(out_h2)*x_2
	pd_w3 = 2*(out_y1 -y_1)*sigmoid_derivative(out_y1)*w_6*sigmoid_derivative(out_h2)*x_1
	pd_w2 = 2*(out_y1 -y_1)*sigmoid_derivative(out_y1)*w_5*sigmoid_derivative(out_h1)*x_2
	pd_w1 = 2*(out_y1 -y_1)*sigmoid_derivative(out_y1)*w_5*sigmoid_derivative(out_h1)*x_1
	print ("partial derivative with respect of : w_6=",pd_w6)
	print ("partial derivative with respect of : w_5=",pd_w5)
	print ("partial derivative with respect of : w_4=",pd_w4)
	print ("partial derivative with respect of : w_3=",pd_w3)
	print ("partial derivative with respect of : w_2=",pd_w2)
	print ("partial derivative with respect of : w_1=",pd_w1)
	W_6 = w_6- alpha*pd_w6
	W_5 = w_5- alpha*pd_w5
	W_4 = w_4- alpha*pd_w4
	W_3 = w_3- alpha*pd_w3
	W_2 = w_2- alpha*pd_w2
	W_1 = w_1- alpha*pd_w1
	print ("bp: W_6=",W_6)
	print ("bp: W_5=",W_5)
	print ("bp: W_4=",W_4)
	print ("bp: W_3=",W_3)
	print ("bp: W_2=",W_2)
	print ("bp: W_1=",W_1)
	# return updated weights and RMSE 
	print("\nlist in the order of: W_1,W_2,W_3,W_4,W_5,W_6,RMSE ")
	return ([W_1,W_2,W_3,W_4,W_5,W_6,RMSE])



def main():
	# initial guess for weights
	w_1 = 0.12 
	w_2 = 0.2
	w_3 = 0.11
	w_4 = 0.25
	w_5 = 0.21
	w_6 = 0.3
	# run the first two trainning to get difference in error
	count=1
	print("Round:",count)
	updated_weight_rmse_pre=forward_and_back_propagation(w_1,w_2,w_3,w_4,w_5,w_6)
	count=count+1
	print("Round:",count)
	updated_weight_rmse_post=forward_and_back_propagation(
		updated_weight_rmse_pre[0],
		updated_weight_rmse_pre[1],
		updated_weight_rmse_pre[2],
		updated_weight_rmse_pre[3],
		updated_weight_rmse_pre[4],
		updated_weight_rmse_pre[5])
	while(abs(updated_weight_rmse_pre[-1] - updated_weight_rmse_post[-1])>threshold):
		print("!!!!!!!iternation:",count)
		print(updated_weight_rmse_pre)
		print(updated_weight_rmse_post)
		print("Difference in error of two rounds of training: ", abs(updated_weight_rmse_pre[-1] - updated_weight_rmse_post[-1]))
		updated_weight_rmse_pre=updated_weight_rmse_post
		count=count+1
		print("Round:",count)
		updated_weight_rmse_post= forward_and_back_propagation(
			updated_weight_rmse_post[0],
			updated_weight_rmse_post[1],
			updated_weight_rmse_post[2],
			updated_weight_rmse_post[3],
			updated_weight_rmse_post[4],
			updated_weight_rmse_post[5])
		print(updated_weight_rmse_post)
		count =count+1

	print("------------------------------------------")
	print("training stoped, from last round:")
	print(updated_weight_rmse_post)

if __name__ == "__main__":

    main()











