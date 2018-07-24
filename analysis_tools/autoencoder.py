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
        self.encoded_layer = 0
        encode_data = Input(shape=(dim,)) 
        decode_data = Input(shape=(encode_dim,))

        #initial expansion layer
        start_dim = int(dim*(1.0 + initial_growth))
        encoded = Dense(start_dim, **kwargs)(encode_data)
        self.layer_dims.append(start_dim)
        self.encoded_layer += 1

        #compression layers
        current_dim = int(start_dim*shrink)
        while current_dim > encode_dim:
            encoded = Dense(current_dim, **kwargs)(encoded)
            self.layer_dims.append(current_dim)
            self.encoded_layer += 1
            current_dim = int(current_dim*shrink)
        
        #final encoding layer
        #encoded = Dense(encode_dim, **kwargs)(encoded)
        encoded = Dense(encode_dim, activation='linear')(encoded) 
        self.layer_dims.append(encode_dim)
        self.encoded_layer += 1
        
        #first expansion layer
        reversed_dims = self.layer_dims[::-1][1:]
        decoded = Dense(reversed_dims[0], **kwargs)(decode_data)
        self.layer_dims.append(reversed_dims[0])
        self.encoded_layer += 1
        
        #remaining expansion layers
        for current_dim in reversed_dims[1:]:
            decoded = Dense(current_dim, **kwargs)(decoded)
            self.layer_dims.append(current_dim)
            
        #final linear layer
        decoded = Dense(dim, activation='linear')(decoded)
        self.layer_dims.append(dim)
        
        #create encoder and decoder models
        self.encoder = Model(encode_data, encoded)
        self.decoder = Model(decode_data, decoded)
        
        #create the joint model
        _encoded_ = self.encoder(encode_data)
        _decoded_ = self.decoder(_encoded_)
        self.autoencoder = Model(encode_data, _decoded_)
        
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
        
    def Encode(self, features, **kwargs):
        return self.encoder.predict(features, **kwargs)
    
    def Decode(self, features, **kwargs):
        return self.decoder.predict(features, **kwargs)
    
    def Summary(self, **kwargs):
        return self.autoencoder.summary(**kwargs)