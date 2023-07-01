import tensorflow as tf
from tensorflow.keras import layers, models

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16
from data_augmentation import train, test, val



batches_per_epoch = len(train)
lr_decay = (1./0.75 -1)/batches_per_epoch
opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay)

class FaceTracker(Model): 
    def __init__(self, eyetracker,  **kwargs): 
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt
    
    def train_step(self, batch, **kwargs): 
        
        X, y = batch
        
        with tf.GradientTape() as tape: 
            classes, coords = self.model(X, training=True)
            
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            
            total_loss = batch_localizationloss+0.5*batch_classloss
            
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        
        opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
    
    def test_step(self, batch, **kwargs): 
        X, y = batch
        
        classes, coords = self.model(X, training=False)
        
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss+0.5*batch_classloss
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
        
    def call(self, X, **kwargs): 
        return self.model(X, **kwargs)

def create_model():
    vgg = VGG16(include_top=False)

    print(vgg.summary())

    def build_model(): 
        input_layer = Input(shape=(120,120,3))
        
        vgg = VGG16(include_top=False)(input_layer)

        # Classification Model  
        f1 = GlobalMaxPooling2D()(vgg)
        class1 = Dense(2048, activation='relu')(f1)
        class2 = Dense(1, activation='sigmoid')(class1)
        
        # Bounding box model
        f2 = GlobalMaxPooling2D()(vgg)
        regress1 = Dense(2048, activation='relu')(f2)
        regress2 = Dense(4, activation='sigmoid')(regress1)
        
        facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
        return facetracker
    facetracker = build_model()

    print(facetracker.summary())

    X, y = train.as_numpy_iterator().next()

    X.shape

    classes, coords = facetracker.predict(X)

    print("Testing data integrity")

    print(classes, coords)

    def localization_loss(y_true, yhat):            
        delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
                    
        h_true = y_true[:,3] - y_true[:,1] 
        w_true = y_true[:,2] - y_true[:,0] 

        h_pred = yhat[:,3] - yhat[:,1] 
        w_pred = yhat[:,2] - yhat[:,0] 
        
        delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
        
        return delta_coord + delta_size

    classloss = tf.keras.losses.BinaryCrossentropy()
    regressloss = localization_loss

    localization_loss(y[1], coords)

    classloss(y[0], classes)

    regressloss(y[1], coords)


        
    print("compiling Model, preparing model for training...")
        
    model = FaceTracker(facetracker)

    model.compile(opt, classloss, regressloss)
    return model, facetracker
model, facetracker = create_model()

if __name__ == "__main__":
    pass
    # model, facetracker = create_model()
    
