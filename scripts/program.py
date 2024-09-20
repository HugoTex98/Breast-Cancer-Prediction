import sys
# Import scripts functions to be used here
from __init__ import (LoadDF, Load, Clear, Describe,
                      Sort, Correlation, Splitscale, SvmModel, 
                      RandomForestModel, AnnModel, Metrics, 
                      create_results_run_folder, logging)


def breast_cancer_prediction_program(dataset_directory: str):
    user_input = 0
    while True:
        try:
            user_input = int(input("Which command you want to use? \n \
            1- LOAD \n \
            2- LOADF \n"))
        except ValueError:
            logging.error("Only integers are accepted!")
            continue
        if user_input >= 1 and user_input <= 2:
            logging.info(f'\nInserted the command: {user_input}')
            break
        else:
            logging.info('The command must be 1 or 2')

    if user_input == 1:
        breast_cancer_data = Load()
        
    elif user_input == 2:
        breast_cancer_data = LoadDF(dataset_directory)


    user_input2 = 0
    while True:
        try:
            user_input2 = int(input("Do you want to use any of these commands? \n \
            1- CLEAR \n \
            2- QUIT \n \
            3- Proceed \n"))
        except ValueError:
            logging.error("Only integers are accepted!")
            continue
        if user_input2 >= 1 and user_input2 <= 3:
            logging.info(f'\nInserted the command: {user_input2}')
            break
        else:
            logging.info('The command must be 1, 2 or 3')

    if user_input2 == 1:
        breast_cancer_dataset = Clear(breast_cancer_data)
        
    elif user_input2 == 2:
        if not breast_cancer_data.empty:
            del breast_cancer_data
            logging.info("DataFrame deleted! Program is over.")
            sys.exit()
        else:
            logging.info("Not possible to delete the DataFrame! Program is over.")
            sys.exit()
        
    elif user_input2 == 3:
        breast_cancer_dataset = breast_cancer_data
        pass


    user_input3 = 0
    # Create folder for store the results of each run
    results_folder = create_results_run_folder()
    while True:
        try:
            user_input3 = int(input("\nWhich command do you want to use for data processing and visualization? \n \
            5- DESCRIBE \n \
            6- SORT \n \
            7- CORRELATION \n \
            8- SPLITSCALE \n \
            9- SVM \n \
            10- RANDOMFOREST \n \
            11- ANN \n \
            12- METRICS \n \
            13- Proceed \n \
            14- None of the previously presented \n"))
            
        except ValueError:
            logging.error("\nOnly integers are accepted! \n")
            continue
        
        if user_input3 >= 5 and user_input3 <= 14:
            logging.info(f'\nInserted the command: {user_input3} \n')
            
            if user_input3 == 5:
                Describe(breast_cancer_dataset, results_folder) 
                
            elif user_input3 == 6:
                cleaned_breast_cancer_dataset = Sort(breast_cancer_dataset)
                
            elif user_input3 == 7:
                cleaner_breast_cancer_dataset = Correlation(breast_cancer_dataset, results_folder)

            elif user_input3 == 8:
                X_train, y_train, X_test, y_test = Splitscale(breast_cancer_dataset, results_folder)
            
            elif user_input3 == 9:
                y_test, y_pred_SVM, accuracy_SVM, modelo_SVM = SvmModel(breast_cancer_dataset, results_folder)
                
            elif user_input3 == 10:
                y_test, y_pred_RF, accuracy_RF, modelo_RF = RandomForestModel(breast_cancer_dataset, results_folder)
                
            elif user_input3 == 11:
                y_test, y_pred_ANN, accuracy_ANN, modelo_ANN = AnnModel(breast_cancer_dataset, results_folder)
                
            elif user_input3 == 12:
                y_pred_SVM, y_pred_RF, y_pred_ANN, y_test = Metrics(breast_cancer_dataset, results_folder)
                
            elif user_input3 == 13:
                break

            elif user_input3 == 14:
                logging.info("\nDon't want any data processing and visualization. Exit the program! \n")
                sys.exit()
        
        else:
            logging.info('\nThe command must be an integer between 5 and 14 \n')

    user_input4 = 0
    while True:
        try:
            user_input4 = int(input("Do you want to use any of these commands? \n \
                                    1- CLEAR \n \
                                    2- QUIT \n \
                                    3- Finish program \n"))
        except ValueError:
            logging.error("Only integers are accepted!")
            continue
        if user_input4 >= 1 and user_input4 <= 3:
            logging.info(f'\nInserted the command: {user_input4}')
            break
        else:
            logging.info('The command must be 1, 2 or 3')

    if user_input4 == 1:
        Clear(breast_cancer_data)
        
    elif user_input4 == 2:
        if not breast_cancer_data.empty:
            del breast_cancer_data
            logging.info("DataFrame deleted! Program is over.")
            sys.exit()
        else:
            logging.error("Not possible to delete the DataFrame! Program is over.")
            sys.exit()
        
    elif user_input4 == 3:
        logging.info('\nProgram finished! \n')
        sys.exit()
