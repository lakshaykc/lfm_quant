## Dataset

Fundamental and Price data can be sourced from various vendors. 
Different vendors have different standards of reporting. 
In this document you will find guidelines to create a module named `dataset.py` 
that should be customized to the reporting style of your data vendor.

The goal of this module is to process the source data into a format that deep-quant accepts.
On a high level, `dataset.py` converts fundamental, price data into sequences for each company
and returns a tensorflow.dataset type object.
 
#### API Notes
* Class name should be `Dataset`

* Processing Steps:
    * Each company has a time series of fundamental and price data. For example, 
    IBM data goes from 1970-01 to 2019-12. Time series are created with a gap of length `stride`. 
    A 12 month stride and time series of length 5 for IBM might look like:
    <br>
    <br>
         Feature Set - [date, tic, feature_1, feature_2, ... feature_n] <br> 
         <br>
         Sequence 1 <br>
         1970-01 IBM 200 220 ... 240 <br>
         1971-01 IBM 201 221 ... 241 <br>
         1972-01 IBM 202 222 ... 242 <br>
         1973-01 IBM 203 223 ... 243 <br>
         1974-01 IBM 204 224 ... 244 <br>
         <br>
         Sequence 2 <br>
         1970-02 IBM 300 320 ... 340 <br>
         1971-02 IBM 301 321 ... 341 <br>
         1972-02 IBM 302 322 ... 342 <br>
         1973-02 IBM 303 323 ... 343 <br>
         1974-02 IBM 304 324 ... 344 <br>
        
        ... and so on. Note that feature values are made up for illustration purposes only.
        
    * Such time series are created for all the companies. 
    
    * The data is centered by mean and scaled by standard deviation for each feature before 
    creating the time series
    
    * Each sequence is normalized by the market cap of the last time step
    
    * More details on data processing can be found in Section 3 of the paper. 


* Required methods:
    * generate_dataset() <br>
    returns tf.dataset object type which is fed into the neural network.
    <br>
    `tf.dataset` should yield a batch of shape `(batch_size, sequence_length, num_features)`
    <br>
    The sequence example above includes non-financial meta features such as `date, tic`. 
    The sequences used in creating the tf.dataset object only contain the features specified in the
    config file.