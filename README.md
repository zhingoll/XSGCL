# XSGCL
## XSGCL: A Lightweight Graph Contrastive Learning Framework for Recommendation

Thank you for your interest in our work. This document is intended to provide detailed guidance on implementing the XSGCL within the [SELFRec framework](https://github.com/Coder-Yu/SELFRec). 
Due to the fact that XSGCL is designed based on SELFRec, in order to respect the work of the original author, only the code related to this work is provided here.

### Key Components

1. **Model Implementation**: The core implementation of the XSGCL model can be found in the `XSGCL.py` file. This file contains the main architecture and algorithmic logic of the model.
2. **Feature Augmentation**: The HLVS feature augmentation method is implemented in the `contrastLoss_ln_var` method within `loss_torch.py`.
3. **Loss Function**: The implementation of the PBPR loss function is provided in the `bpr_loss_pop` method within `loss_torch.py` and the `pop` method in `XSGCL.py`. 

### Configuration and Execution

To run the XSGCL model within the SELFRec framework, please follow these steps:

1. **Environment Setup**: Ensure that your runtime environment meets the dependencies required by SELFRec.
2. **Model Configuration**: Relevant hyperparameters for the model can be found and adjusted in the `XSGCL.conf` file.
3. **Run the Model**: Once the model is configured according to the guidance in the SELFRec documentation, it is ready to be executed.

### Please Note

When using or referencing this code, please adhere to the corresponding open-source license.

### Contact Us

If you encounter any issues while using the model, or if you would like to discuss technical details with me, please feel free to contact me through the issues.
