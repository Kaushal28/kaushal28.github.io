---
layout: post
excerpt: Don't worry, it's easier than it looks
comments: true
images:
  - url: /assets/MultiOutputCNN.png
---
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        inlineMath: [['$','$']]
      }
    });
  </script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<br>
<p> Recently I participated in a Kaggle computer vision competition which included multi-label image classification problem. Here is the link to Kaggle competition: <a href='https://www.kaggle.com/c/bengaliai-cv19'> https://www.kaggle.com/c/bengaliai-cv19 </a> </p>

<p> Here’s a brief description about the competition: We were supposed to classify given Bengali graphemes components (similar to English phonemes) into one of 186 classes (168 grapheme root, 11 vowel diacritics and 7 consonant diacritics). Every image will have three components and we were supposed to identify these three components in the given image. </p>

<p> So as you can see, this is a multi-label classification problem (Each image with 3 labels). To address these type of problems using CNNs, there are following two ways: </p>

<ul>
<li> Create 3 separate models, one for each label. </li>
<li> Create a single CNN with multiple outputs. </li>
</ul>
  
<p> Let’s first see why creating separate models for each label is not a feasible approach. When we create separate models, almost all the layers will be the same except the last one or two layers. So the training time will be very high (if a single model takes $x$ time, then $n$ separate models will take $n * x$ time). </p>

<p> Now let’s explore CNN with multiple outputs in detail. </p>

<p> Here is high level diagram explaining how such CNN with three output looks like: </p>

<img src="/assets/MultiOutputCNN.png">

<p> As you can see in above diagram, CNN takes a single input `X` (Generally with shape (m, channels, height, width) where m is batch size) and spits out three outputs (here Y2, Y2, Y3 generally with shape (m, n_classes) again m is batch size). </p>

<p> For each output, we can specify a separate name, callback function (for example learning rate annealer), activation function, even the loss function and metrics. Now let’s see how to implement all these using Keras. </p>

<p> Let’s first create a basic CNN model with a few Convolutional and Pooling layers. </p>

{% highlight python %} 
inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 1))  # Considering gray scale images

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1))(inputs)
model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = Dropout(rate=0.3)(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=256, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = Dropout(rate=0.3)(model)

# A few flatten layers
model = Flatten()(model)
model = Dense(1024, activation = "relu")(model)
model = Dropout(rate=0.3)(model)
dense = Dense(512, activation = "relu")(model)

# Finally what you are waiting for: Multiple outputs!
head_root = Dense(168, activation = 'softmax')(dense)  # Here 168 is the number of classes.
head_vowel = Dense(11, activation = 'softmax')(dense)
head_consonant = Dense(7, activation = 'softmax')(dense)

model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])
{% endhighlight %}

<br>

<div id="disqus_thread"></div>
<script>

/**
*  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
*  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables*/
/*
var disqus_config = function () {
this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
};
*/
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://kaushal28-github-io.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>

<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-162164038-1"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'UA-162164038-1');
</script>
