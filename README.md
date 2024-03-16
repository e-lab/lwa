plan is to train the model is a two step approach 

IDEA 1
step 1 predict what the Nth scene would be. Nth scene should be the next most disimilar scene from what the model has just seen this comparision should be done in the concept space (latent space) 
step 2 after the model has learned to predict the Nth scene alow it to generate an action sequence to move itself to match what it this will happen in the nth scene. The comparision should be between the ground truth nth scene and a live play of the models actions done in the concept space (latent space) or because we should have the RGB values of both comparisions in the real world space. 

IDEA 2 is that we can skip step 1 and only perform step 2 for end to end training. this would alow the model to learn anything. 
IDEA 3 is we should do end to end training but find a way to difuse the cs1 and actions without live play to speed up training such that the ground truth and model results can be compaired directly allowing us to automate pipeline wihtout additional add ons 


