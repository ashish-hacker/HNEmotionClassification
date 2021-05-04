import pickle
import streamlit as st
#import FindMyMood as f 
import text2emotion as te

# loading the trained model
#pickle_in = open('nlp_model.pkl', 'rb') 
#classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
# this is the main function in which we define our webpage  
def prediction(sentence):
	custom_tokens = f.remove_noise(f.word_tokenize(sentence))
	
	result = classifier.classify(dict([token, True] for token in custom_tokens))
	return result
	
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:orange;padding:13px"> 
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
        #result = prediction(text)
        result_dict = te.get_emotion(text)
        result_value = 0
        for key, value in result_dict.items():
        	if value > result_value:
        		result = key
        if result_value == 0:
        	result = "Neutral"
        st.success('Your emotion is {}'.format(result))
        print(result)
     
if __name__=='__main__': 
    main()
