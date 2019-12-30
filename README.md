# EE838A_HW2
Homework 2, Advanced Image Restoration and Quality Enhancement, EE, KAIST, Fall 2018

1. Training
	- Copy the HDR training images into the folder './data/train/HDR'
	- In the folder './source', open Command Prompt if using Window or Terminal if using Linux
	- Type: python main.py --mode=train
	- Wait about 1.5 hours to finish training

	+ The weights of model are saved in './model'
	+ All logs are written in the file './logs/logs_train.txt'

2. Validation
	- If you want to use my best model, delete all files in the foldler './model', then copy all files in the folder './report/best_model' to './model'
	- If you want to use different validation dataset:
		> delete all files in two folder './data/valid/HDR' and './data/valid/LDR'
		> copy the new validation HDR images to './data/valid/HDR'
		> copy the new validation LDR images, which must have format PNG, to './data/valid/LDR'
		> all the new validation images must be renamed following format Cxx_HDR.hdr and Cxx_LDR.png, where xx is the index of each image. For example: C16_HDR.hdr, C16_LDR.png

	- In the folder './source', open Command Prompt if using Window or Terminal if using Linux
	- Type: python main.py --mode=valid
	- Wait until finish validation

	+ The validation results are stored in the folder './report/valid_result'
	+ All logs are written in the file './logs/logs_valid.txt'

3. Test
	- If you want to use my best model, delete all files in the foldler './model', then copy all files in the folder './report/best_model' to './model'
	- All testing LDR images must be same PNG format and also same size, for example MxN
	- In the folder './source', open Command Prompt if using Window or Terminal if using Linux
	- Type: python main.py --mode=test --img_height=M --img_width=N
	- Wait until finish testing

	+ The testing results are stored in the folder './report/test_result'
	+ All logs are written in the file './logs/logs_test.txt'

4. Quality evaluation
	- Open Matlab
	- Install the HDR Toolbox [2] in Matlab
	- Copy HDR-VDP-2.2.1 to the folder'./hdrvdp'
	- Browse the work space of Matlab to the folder './source'
	- To calculate mPSNR of the validation images, in the Command Window of Matlab, type: calc_mpsnr
	- To display visualization map and determine quality scores, in the Command Window of Matlab, type: calc_hdr_vdp

	+ In line 5 and 12 of the file './source/calc_hdr_vdp.m', change the link of reference and distorted image

5. Explaination
	- Read './report/HW2_20184187_DinhVu_Report.pdf' for the detail explaination

6. Reference

[1] Gabriel Eilertsen, Joel Kronander, Gyorgy Denes, Rafal K. Mantiuk and Jonas Unger, “HDR image reconstruction from a single exposure using deep CNNs”, ACM Transactions on Graphics, volume 36, no. 6, article 178, October 2017.

[2] Francesco Banterle, Alessandro Artusi, Kurt Debattisa and Alan Chalmers, “Advanced High Dynamic Range Imaging 2nd Edition”, AK Peters (CRC Press), July 2017.

[3] Rafal Matiuk, Kil Joong Kim, Allan G. Rempel and Wolfgang Heidrich, “HDR-VDP-2: A calibrated visual metric for visibility and quality predictions in all luminance conditions”, ACM Transactions on Graphics, volume 30, issue 4, article no. 40, July 2011.

