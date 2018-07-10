from keras.layers import Input, Dense
from keras.models import Model

#this is essentially a wrapper for keras that automates the building of an autencoder
#based on the initial dimensionality, target dimensionality, and how fast the layer 
#dimensions should shrink. this also includes an initial pre-expansion as is generally
#a good idea for autoencoders
class AutoEncoder:
    
    #determine the dimensions of the autoencoder (i.e., how big each layer is)
    def __init__(self, dim, encode_dim, 
                 initial_growth=0.10, shrink=0.75,
                 **kwargs):
        
        self.layer_dims = []
        data = Input(shape=(dim,))       

        #initial expansion layer
        start_dim = int(dim*(1.0 + initial_growth))
        encoded = Dense(start_dim, **kwargs)(data)
        self.layer_dims.append(start_dim)

        #compression layers
        current_dim = int(start_dim*shrink)
        while current_dim > encode_dim:
            encoded = Dense(current_dim, **kwargs)(encoded)
            self.layer_dims.append(current_dim)
            current_dim = int(current_dim*shrink)
        
        #final encoding layer
        encoded = Dense(encode_dim, **kwargs)(encoded) 
        self.layer_dims.append(encode_dim)
        
        #first expansion layer
        reversed_dims = self.layer_dims[::-1][1:]
        decoded = Dense(reversed_dims[0], **kwargs)(encoded)
        self.layer_dims.append(reversed_dims[0])
        
        #remaining expansion layers
        for current_dim in reversed_dims[1:]:
            decoded = Dense(current_dim, **kwargs)(decoded)
            self.layer_dims.append(current_dim)
            
        #final linear layer
        decoded = Dense(dim, activation='linear')(decoded)
        self.layer_dims.append(dim)
        
        #save the models
        self.autoencoder = Model(data, decoded)
        self.encoder = Model(data, encoded)
        
        return None
    
    def Compile(self, **kwargs):
        return self.autoencoder.compile(**kwargs)
    
    def Fit(self, features_train, features_test, **kwargs):
        if 'validation_data' in kwargs:
            return self.autoencoder.fit(features_train, 
                                        features_train, 
                                        **kwargs)
        else:
            return self.autoencoder.fit(features_train, 
                                        features_train,  
                                        validation_data=(features_test, features_test), 
                                        **kwargs)
        
    def Predict(self, features, **kwargs):
        return self.encoder.predict(features, **kwargs)
    
    def Summary(self, **kwargs):
        return self.autoencoder.summary(**kwargs)