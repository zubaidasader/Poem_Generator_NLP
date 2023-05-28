# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np
# import pickle

# # Load the tokenizer
# with open('en_tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# # Load the saved model
# model = tf.keras.models.load_model("en_poem_generation_model.h5")

# max_sequence_len = model.layers[0].input_shape[1]

# def generate_poem(seed_text, next_words, temperature):
#     output_text = seed_text
#     for _ in range(next_words):
#         token_list = tokenizer.texts_to_sequences([seed_text])[0]
#         token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
#         predicted_probs = model.predict(token_list, verbose=0)[0]
#         predicted_probs = np.log(predicted_probs) / temperature
#         predicted_probs = np.exp(predicted_probs) / np.sum(np.exp(predicted_probs))
#         predicted_index = np.random.choice(len(predicted_probs), size=1, p=predicted_probs)[0]
#         output_word = tokenizer.index_word[predicted_index]
#         seed_text += " " + output_word
#         output_text += " " + output_word
#     return output_text

# def main():
#     st.title("Poem Generator")
#     seed_text = st.text_input("Enter the seed text:", "Happiness is")
#     next_words = st.slider("Select the number of words to generate:", 10, 100, 25)
#     temperature = st.slider("Select the temperature:", 0.1, 1.0, 0.6, step=0.1)

#     if st.button("Generate Poem"):
#         output_text = generate_poem(seed_text, next_words, temperature)
#         st.success(output_text)

# if __name__ == "__main__":
#     main()
import streamlit as st
import gpt_2_simple as gpt2

def generate_text(seed_text, length, temperature, run_name):
    text = gpt2.generate(sess, length=length, temperature=temperature, prefix=seed_text, run_name=run_name, return_as_list=True)[0]
    return text

def main():
    st.title("Poems Generator")
    seed_text = st.text_input("Enter the seed text:", "Happiness is")
    length = st.slider("Select the number of words to generate:", 10, 100, 25)
    temperature = st.slider("Select the temperature:", 0.1, 1.0, 0.6, step=0.1)

    if st.button("Generate Text"):
        generated_text = generate_text(seed_text, length, temperature, "run2")
        st.success(generated_text)

if __name__ == "__main__":
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name="run2")
    main()