import pickle
import streamlit as st
import FindMyMood as f 

# loading the trained model
pickle_in = open('nlp_model.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Emotion Classification ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    text = st.text_input("Input your text to classify")
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        tokens = f.remove_noise(f.word_tokenize(text))
        result = classifier.classify(dict([token, True] for token in custom_tokens))
        st.success('Your emotion is {}'.format(result))
        print(result)
     
if __name__=='__main__': 
    main()