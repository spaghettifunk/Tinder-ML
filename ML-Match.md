# General Idea
* Machine learning based dating
* Different focus possible
* Maybe by using APIs of services (Tinder/Facebook)
* the algorithm should find good matches for the person searching for a date
* How to test it? - Get all friends to use it and rate it for science?
* How to match eye- and hair-color
 
# Different Focus for Machine Learning
## Picture matching
* Eigenface
* What possible algorithms to use
* How to learn what a person likes and not
	* Hot or not?
* could be used against Tinder
* Categories? (how to find the posh persons, the hippies, ..., nose, cheeks, chin, face proportions, ...)

### Goals

## Property matching
* Matching more standard values
* Hobbies, Interests, Characteristics
* Find the special blend
* More like OkCupid but with Machine Learning Optimisation

### Goals

## Language analysis
* Analyse beginnings of Tinder conversations
* What works, what not
* Need big data pool for analysing that.

### Goals

## The bigger picture
* Automated Tinder
* Blinddate-Bot with Input/Friends input
* How to evaluate data, how to find out if it works?

# Previous Work
* <http://robrhinehart.com/?p=1005>
* <http://www.fastcolabs.com/3028414/how-facebooks-machines-got-so-good-at-recognizing-your-face>
* <https://en.wikipedia.org/wiki/Eigenface>
* [Face Database](http://www.face-rec.org/databases/)

## Tinderbot
* <http://techcrunch.com/2015/02/10/tinderbox/>
* <https://github.com/crockpotveggies/tinderbox>
* <http://crockpotveggies.com/2015/02/09/automating-tinder-with-eigenfaces.html>
* written in Scala
* Using Eigenfaces
* We should look at it first
 
# Papers
## Face recognitions
* <http://sightcorp.com/downloads/Machine%20Learning%20Techniques%20for%20Face%20Analysis.pdf>
 
# Possible tools
## Face recognition
* [OpenCV](https://en.wikipedia.org/wiki/OpenCV)
	* Can do Image Processing, Feature Detection and also has some Machine Learning Capabilities (K-Nearest Neighbour, Support Vector Machines)
	* Supports Eigenfaces, Fisherfaces
* Support Vector Machines seem to be used often in general
* [A Talk about Face Recognition and Python](https://www.youtube.com/watch?v=lJYEup-0gJo)
* <https://pypi.python.org/pypi/facereclib>
	* A lots of algorithms
* Another face recognition library <https://github.com/bytefish/facerec>
* scikit-learn also seems to support Eigenfaces and SVM
* Eigenfaces is supervised learning

## Property matching
* more classical approach
* Easy to define classes
* Reinforcement learning?
* scikit-learn
* pybrain
* using OkCupid API?
