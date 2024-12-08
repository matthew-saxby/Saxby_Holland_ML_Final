import sys
import os
module_path = os.path.join(os.path.dirname(__file__), 'src', 'models')
sys.path.append(module_path)
from train_model import main


#calls main function.
main()