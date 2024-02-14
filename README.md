# Chest X-ray Disease Detection

## Project Abstract
Our project, the Chest X-ray Disease Detection, has a dual objective. Firstly, it aims to offer an advanced solution for identifying abnormalities in chest X-rays. Secondly, it introduces a unique approach to improve accuracy by integrating existing methods with new layers and extensive training. As healthcare professionals and medical facilities as our target audience; Determining whether a person has a disease is extremely critical. We intend to utilize this approach as a supplementary measure rather than a substitute.
 
## High-Level Requirement
**User Needs and High-Level Requirements:**
* Target Users:
Healthcare Professionals: Radiologists, and healthcare providers who require quick and accurate chest X-ray results.
Medical Facilities: Hospitals, clinics, and diagnostic centers seeking an efficient tool for disease detection in chest X-rays.
* User Interface (UI):
Easy-to-Navigate Dashboard: A user-friendly interface with options to upload chest X-ray images for analysis.
Results Display: Clear presentation of analysis results, including disease classification and confidence scores.
* Functionality:
Automated Analysis: The system should automatically process and analyze chest X-ray images without manual intervention.
Disease Detection: Accurately identify and classify chest X-ray abnormalities, including Pneumonia, COVID-19, and Tuberculosis.
Speed and Efficiency: Provide rapid results to support timely decision-making in clinical settings.

 
## Conceptual Design
**Software Architecture:**
Backend Framework: robust backend using Python and popular web frameworks like Django or Flask.
Machine Learning Frameworks: Utilize machine learning libraries like TensorFlow or PyTorch for model development and deployment.
Frontend Interface: a user-friendly web-based interface using HTML, CSS, and JavaScript for image upload and result display.

**Programming Languages:**
Backend: Python will be the primary language for developing the backend logic, including data processing, machine learning model integration, and API creation.
Frontend: HTML, CSS, and JavaScript for designing the user interface and handling user interactions

## Background
There has been existing work to determine whether a person has Pneumonia, Covid-19, or Tuberculosis through Chest X-Ray. As accuracy in the medical field is extremely important, we want to improve the existing methods by exploring more models and handling models on various splits of datasets to calculate the accuracy when overfitting occurs to explore the learning pattern from other classes.
Furthermore, we want to explore whether we can determine various types of Pneumonia as well as add another dataset to train further after the first batch is completed to see whether it has significant improvement.

## Collaborators

[//]: # ( readme: collaborators -start )
<table>
<tr>
    <td align="center">
        <a href="https://github.com/thanhnguyen46">
            <img src="https://avatars.githubusercontent.com/u/60533187?v=4" width="100;" alt="Thanh Nguyen"/>
            <br />
            <sub><b>Thanh Nguyen</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/manan305">
            <img src="https://avatars.githubusercontent.com/u/159965792?v=4" width="100;" alt="Thanh Nguyen"/>
            <br />
            <sub><b>Thanh Nguyen</b></sub>
        </a>
    </td>
</tr>
</table>

[//]: # ( readme: collaborators -end )