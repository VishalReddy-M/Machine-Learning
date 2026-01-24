#BAGGING
'''it is also known as bootstrap aggregating
it means training many models instead of one and combining their answers to get a better and more stable result'''
''' it goes on majority of voting in classification(yes/no)
    it goes on average of voting in regression (house price predictions)'''
'''SYNTAX:
from sklearn.ensemble import RandomForestClassification
model=RandomForestClassification(parameters....,bootstrap=True)'''
#BOOSTING
'''it is also an ensemble learning technique where models are trained one after another each new model focuses on 
correcting the mistake made by the previous model'''
'''it is also known as sequential model'''
#RANDOM FOREST ALGORITHM
#parameters:
'''1.n_estimators
    no of trees(100-300--> balanced) default:100
   2.max_depth
    it says depth of trees(small(10->20) is preferred),default:None,depends on dataset size
   3.min_samples_split
    min samples needed to split a node,default:2,limit:int>=2
   4.min_samples_leaf:
    min samples of leaf node,default:1,limit:int>=1
   5.bootstrap
    whether sampling should follow the randomness and bagging ,default:True Mandatory
   6.criterion
    classification:gini,entropy
    regression:squared_error,poission
   7.random_state
    default:none,limit:int>=40     '''