from keras.layers import Input, Dense, concatenate, Lambda, Average
from keras.models import Model

#this is essentially a wrapper for keras that automates the building of an autencoder
#based on the initial dimensionality, target dimensionality, and how fast the layer 
#dimensions should shrink. this also includes an initial pre-expansion as is generally
#a good idea for autoencoders
class AutoEncoder:
    
    #determine the dimensions of the autoencoder (i.e., how big each layer is)
    def __init__(self, dim, encode_dim, num_clones,
                 initial_growth=0.10, shrink=0.75,
                 **kwargs):
        
        self.layer_dims = []
        self.encoded_layer = 0
        encode_data = Input(shape=(dim,)) 
        decode_data = Input(shape=(encode_dim,))
        
        
        ################################################
        ###build the encoder and decoder architecture###
        ################################################

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
        encoded = Dense(encode_dim, **kwargs)(encoded)
        #encoded = Dense(encode_dim, activation='sigmoid')(encoded) 
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
        
        
        #########################################
        ###build the coupled autoencoder model###
        #########################################
        
        #################
        ###the encoder###
        #################
        
        #read in joint data and split into the columns
        joint_encode_data = Input(shape=(dim*num_clones,))
        split_encode_data = []
        for i in range(num_clones):
            split_encode_data.append( Lambda(lambda x : x[:,i*dim:(i+1)*dim], output_shape=(dim,))(joint_encode_data) )
            
        #output from tied encoders
        split_encoded = [self.encoder(input) for input in split_encode_data]
        
        #average the output from each encoder
        average_encoded = Average()(split_encoded)
        
        #create the full encoder
        self.joint_encoder = Model(joint_encode_data, average_encoded)
        
        
        #################
        ###the decoder###
        #################
        
        average_encoded = Input(shape=(encode_dim,))
            
        #spit out the results and concatenate
        split_decoded = [self.decoder(average_encoded) for i in range(num_clones)]
        if num_clones == 1:
            joint_decode_data = split_decoded[0]
        else:
            joint_decode_data = concatenate(split_decoded, axis=-1)
        
        #final model
        self.joint_decoder = Model(average_encoded, joint_decode_data)
        
        
        #####################
        ###the autoencoder###
        #####################

        encoded = self.joint_encoder(joint_encode_data)
        joint_decode_data = self.joint_decoder(encoded)
        self.joint_autoencoder = Model(joint_encode_data, joint_decode_data)
        
        return None
    
    def Compile(self, **kwargs):
        return self.joint_autoencoder.compile(**kwargs)
    
    def Fit(self, features_train, features_test, **kwargs):
        if 'validation_data' in kwargs:
            return self.joint_autoencoder.fit(features_train, 
                                        features_train, 
                                        **kwargs)
        else:
            return self.joint_autoencoder.fit(features_train, 
                                        features_train,  
                                        validation_data=(features_test, features_test), 
                                        **kwargs)
        
    def Encode(self, features, **kwargs):
        return self.joint_encoder.predict(features, **kwargs)
    
    def Decode(self, features, **kwargs):
        return self.joint_decoder.predict(features, **kwargs)
    
    def Summary(self, **kwargs):
        return self.joint_autoencoder.summary(**kwargs)