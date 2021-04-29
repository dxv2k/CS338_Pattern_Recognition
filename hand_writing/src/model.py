import tensorflow as tf

class SimpleCNN(tf.keras.models): 
    def __init__(self, data): 
        '''
        '''
        self.model = None
        self.data = data
        self.optimizer = None


    def build_model(self, input, output): 
        ''' 
        ''' 
        pass 

    def train(self, data_train, label_train): 
        ''' 
        ''' 
        pass 

    def predict(self): 
        pass 

    def summary(self): 
        self.model.summary() 
    
    def save_model(self): 
        pass 

    def load_model(self): 
        pass