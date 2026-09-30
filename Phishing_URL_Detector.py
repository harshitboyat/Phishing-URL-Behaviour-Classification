#--------------------------------------------------
# PHISHING URL BEHAVIOUR CLASSIFICATION
#              STREAMLIT APP
#--------------------------------------------------

#importing the libraries
import streamlit as st
import pandas as pd
import joblib

from urllib.parse import urlparse
import ipaddress

#------------------------------------------
#        Loading Trained Model
#------------------------------------------
model = joblib.load("phishing_url_model.pkl")
scaler = joblib.load("phishing_url_scaler.pkl")
feature_names = joblib.load("phishing_url_features.pkl")

print(feature_names)
print("Total Features:", len(feature_names))

#------------------------------------------
#        URL Feature Extraction
#------------------------------------------

def extract_url_features(url):

    # Adding scheme if it is not provided
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # Parsing the URL
    parsed_url = urlparse(url)

    # Getting the domain
    domain = parsed_url.hostname

    if domain is None:
        domain = ""

    # Getting the path and query
    path = parsed_url.path
    query = parsed_url.query

    # Complete URL
    complete_url = url

    #------------------------------------------
    #        Basic URL Features
    #------------------------------------------

    URLLength = len(complete_url)

    DomainLength = len(domain)

    # Checking whether domain is an IP address
    try:
        ipaddress.ip_address(domain)
        IsDomainIP = 1
    except ValueError:
        IsDomainIP = 0

    #------------------------------------------
    #        TLD and Subdomain
    #------------------------------------------

    domain_parts = domain.split(".")

    if len(domain_parts) >= 2:
        TLD = domain_parts[-1]
        TLDLength = len(TLD)
    else:
        TLDLength = 0

    if len(domain_parts) > 2:
        NoOfSubDomain = len(domain_parts) - 2
    else:
        NoOfSubDomain = 0

    #------------------------------------------
    #        Character Features
    #------------------------------------------

    NoOfLettersInURL = sum(character.isalpha() for character in complete_url)

    NoOfDegitsInURL = sum(character.isdigit() for character in complete_url)

    if URLLength > 0:
        DegitRatioInURL = NoOfDegitsInURL / URLLength
    else:
        DegitRatioInURL = 0

    #------------------------------------------
    #        Special Character Features
    #------------------------------------------

    NoOfEqualsInURL = complete_url.count("=")

    NoOfQMarkInURL = complete_url.count("?")

    NoOfAmpersandInURL = complete_url.count("&")

    # Special characters other than letters and digits
    special_characters = sum(
        not character.isalnum()
        for character in complete_url
    )

    NoOfOtherSpecialCharsInURL = (
        special_characters
        - complete_url.count(".")
        - complete_url.count("-")
        - complete_url.count("/")
        - complete_url.count("?")
        - complete_url.count("=")
        - complete_url.count("&")
    )

    if URLLength > 0:
        SpacialCharRatioInURL = special_characters / URLLength
    else:
        SpacialCharRatioInURL = 0

    #------------------------------------------
    #        HTTPS Feature
    #------------------------------------------

    if parsed_url.scheme == "https":
        IsHTTPS = 1
    else:
        IsHTTPS = 0

    #------------------------------------------
    #        Creating Feature DataFrame
    #------------------------------------------

    features = {
        "URLLength": URLLength,
        "DomainLength": DomainLength,
        "IsDomainIP": IsDomainIP,
        "TLDLength": TLDLength,
        "NoOfSubDomain": NoOfSubDomain,
        "NoOfLettersInURL": NoOfLettersInURL,
        "NoOfDegitsInURL": NoOfDegitsInURL,
        "DegitRatioInURL": DegitRatioInURL,
        "NoOfEqualsInURL": NoOfEqualsInURL,
        "NoOfQMarkInURL": NoOfQMarkInURL,
        "NoOfAmpersandInURL": NoOfAmpersandInURL,
        "NoOfOtherSpecialCharsInURL": NoOfOtherSpecialCharsInURL,
        "SpacialCharRatioInURL": SpacialCharRatioInURL,
        "IsHTTPS": IsHTTPS
    }

    return pd.DataFrame([features])


#------------------------------------------
#        Streamlit Page
#------------------------------------------

st.title("Phishing URL Behaviour Classification")

st.write("Enter a URL to check whether it is legitimate or phishing.")

url = st.text_input("Enter URL").strip()

check_button = st.button("Check URL")

#------------------------------------------
#        URL Prediction
#------------------------------------------

if check_button:

    if url == "":
        st.warning("Please enter a URL.")

    else:

        #------------------------------------------
        #        Extracting URL Features
        #------------------------------------------

        url_data = extract_url_features(url)

        # Arranging features in model order
        url_data = url_data[feature_names]

        #------------------------------------------
        #        Showing Extracted Features
        #------------------------------------------

        st.write("Extracted URL Features")
        st.write(url_data)

        #------------------------------------------
        #        Scaling URL Features
        #------------------------------------------

        scaled_data = scaler.transform(url_data)

        #------------------------------------------
        #        Making Prediction
        #------------------------------------------

        prediction = model.predict(scaled_data)

        print("*"*60)
        print("Extracted URL Features")
        print("*"*60)

        print(url_data)

        print("*"*60)
        print("Prediction")
        print("*"*60)

        print(prediction)

        #------------------------------------------
        #        Displaying Result
        #------------------------------------------

        if prediction[0] == 1:
            st.success("The URL is classified as Legitimate.")

        else:
            st.error("The URL is classified as Phishing.")