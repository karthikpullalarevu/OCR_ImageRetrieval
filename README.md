# OCR_ImageRetrieval


<h2>Task 1 : OCR</h2>
<br><br>
     <img src = "results/Task1/Memo/google/2000621910.jpg" alt="3d2">
     <br><br>
<details>
<summary>Deployed Asset Usage: </summary> 
    
   <p>
     <br><br>
     1. To perform OCR and get the response image: 
     ```bash
      curl --location 'http://43.205.49.236:6000/predict/readDocument' \
      --form 'image=@"/Users/karthik/Downloads/506888300_506888301.jpg"' \
      --form 'ocr_engine="azure"'
      ```
     <br><br>
     2. To download a JSON with the extractions and NER/POS tags: (change the ocr_engine: 'google', 'azure')
     ```bash
      curl --location 'http://43.205.49.236:6000/predict/getJSON' \
      --form 'image=@"/Users/karthik/Downloads/506888300_506888301.jpg"' \
      --form 'ocr_engine="azure"'
      ```
     <br><br>
   </p>
      
</details>
