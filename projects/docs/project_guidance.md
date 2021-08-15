# Daily guide for projects

## Summary

This project plan explicitly encourages the iterative nature of research as a series of questions and answers that gradually refine your hypotheses. You will start by choosing a topic on day 1, based on your broad research interests (neuroscience, computer vision, natural language processing or reinforcement learning). You will then try to come up with an interesting question on days 2-3 both via brainstorming and via searching the literature for interesting papers. For the rest of week 1 you will look for an appropriate dataset to answer your question, and try to process that dataset into a format that is good for modelling.

In the second week you will experiment with different types of deep neural networks to try to answer your question. During D11, you will know enough to be able to write a short abstract about your project, which may or may not include results, but it should at least include a testable hypothesis. For the rest of the project days on D12-D14 you will focus on getting evidence for/against your hypothesis. Finally, in D15 you will meet with other groups and tell them the story of your project. This is a low-key presentation that may include some of the plots you made along the way, but it is not meant as a real research presentation with high “production values”. See some of the [examples](https://compneuro.neuromatch.io/projects/docs/project_2020_highlights.html) from the comp-neuro course last year to get a sense of what the presentation should look like.

## Project templates

[Project templates](https://deeplearning.neuromatch.io/projects/docs/projects_overview.html) are research ideas developed by the NMA team or by the project TAs which contain code and slides with background and questions. Project templates can be used in a variety of ways.
* For starters, you can use the project templates just to get familiarized with some neural network applications. They can provide keywords for you to use in your literature search, or python libraries you can reuse to answer your own questions.
* You may start with a project template, use it in the first week and then in the second week diverge from it as your group develops their own new idea or question to test.
* Templates have a natural flow of questions, but don’t hesitate to skip or completely change some of these. They are meant to be used very flexibly!

## Project TAs

Project TAs are a new role at NMA this year, and they are your friendly topic experts to consult with on all issues related to that topic. They can also help with other aspects of a project, including brainstorming, literature searches and coding. You will have one-hour meetings with an assigned project TAs starting Tuesday and continuing every 2-3 days until the end of the school. During this time, they will help you refine your question and hypothesis into something that can be answered with deep learning. Sometimes they may miss a meeting or arrive late (busy schedules, lots of meetings!). In those cases, please stop what you were doing to have the meeting, and then resume your work when the project TA leaves.

Sometimes the project TAs might need to schedule meetings slightly earlier or later. Please try to be flexible, they are doing a lot of "virtual footwork" to make sure all the groups have enough support during projects. We also encourage you to reach out to them for extra meetings whenever you need them, and to post questions on discord in the #topic channels. All project TAs have time set aside specifically to answer discord questions and to provide additional meetings when necessary.

## Project Day 1 (Monday, 3 h)

Please submit [this form](https://airtable.com/shrZ341XOB4CSO9wh) at the end of the project period, otherwise you won't be matched to a project TA! Only one form per group is needed, and there is should be two groups in your pod.

Goal is to decide what dataset/topic to work on and to form two separate groups (3-6 students/group). TAs make a [Miro](http://miro.com) concept board for brainstorming at the beginning of this session and they invite the students to join during the brainstorming times below.
* Introductions (30 min = 2 min/student): say a few things about yourself, then about your research area or research interests. What are you really curious about, that you might explore in your NMA project?
* Listen carefully as others talk about their interests. If you are curious about something, ask them. If their interests match yours, you can try to recruit them to your group later!
* Individual reading time (30 min): browse the projects booklet which includes this guide (skim the entire thing) + 16 project templates with slides and code + docs with further ideas and datasets
* Brainstorm at the pod level (60 min): do you have an idea or a rough direction? Bring it out to the group. Start recruiting others to your group!
* The TA will pay close attention to what you all are interested in, and will produce an initial set of two topics / datasets / ideas, written down on the Miro board. Join one of these by writing your name down underneath the topic.    
* Separate into groups. Now brainstorm within your group (60 min, breakout rooms). Definitely choose a topic and start thinking about concrete questions if you can. Make sure the topic you choose is well suited for answering the broad range of questions you are interested in. Try to come up with one or a few topics of interest, either by yourselves or directly from the booklet (i.e. project templates).

Tips:
* No need to have a very concrete project after this day. You will determine the feasibility of your questions in the next few days, and you will likely change your question completely. That’s how research works!  

## Project Day 2 (3h)

* First, check your inbox! You will receive project TA assignments at some point between Monday and Tuesday, with a first meeting scheduled for Tuesday. Reach out to your project TA to confirm the first meeting or reschedule if it’s necessary.

Watch the modelling steps 1-2 video and do the tutorial. This will ask you to come up with a question (today) and do a literature review (tomorrow). In practice, these two steps are intertwined. You might want to get a headstart today on looking at some literature, because it can inform and change your question.

* You will need to use your own project for some of this content. If you don’t yet have concrete ideas, or you haven’t done a research project before, use one of the provided project templates.
* If you are using a project template, your goal is to translate the information from the slide and colab notebook into the 10-steps format. Some information might not be readily available in the slide or notebook, and you might have to find it in your literature review later this day.
* Try to write down a few sentences for each of the two steps applied to your project. Putting thoughts into well-defined sentences and paragraphs helps at all stages of a project.


## Project Day 3 (3h)

Today you should do a literature review (step 2 of the "modelling steps"). The goal of the literature review is to situate your question in context and help you acquire some keywords to help you refine your question.
* (30min) on your own, start doing a literature review using google searches and only look at abstracts to select 2-3 promising ones.  
* (10min) report to the whole group what papers you found and pool them together. Assign one paper per person to read/skim in the next 1h.  
* (1h) on your own, read the paper that was assigned to you. Make notes in a common google doc shared with your group, and especially write down important keywords or concepts which you might use in your proposal later today. If you are not connected to an .edu domain or a VPN, try to find full versions of papers on preprint servers like arXiv / bioRxiv. You could also ask your TA to get it for you (and they might in turn ask someone who has access to a university VPN). There might be other options too…  
* (1h) report back to the group, and try to tell them as much as you understood about the paper. Get into details, but don’t just read to them whole sections from the paper. Ask the other students questions about the papers they are presenting to understand them better.

## Project Days 4-5 (3h/day)

These correspond to "modelling steps 3-4". As usual, watch the short video and start through the notebook. It will ask you to find the ingredients for your project. In our case, this primarily involves finding a dataset. Finding a good dataset is a crucial step in all deep learning projects. Since you already have a question, you need the right dataset for that question. If you can find a good one, it will make the rest of the project much easier, so you should spend a lot of time on this (both days). This goes hand in hand with formulating a hypothesis (step 4), because some datasets will naturally invite you to make a hypothesis, especially if you already have a well-defined question. Here's an example:

Question:  What is the relation between pollution and climate change?  
Dataset:   1) Geographic dataset of pollution indices; 2) Geographic dataset of average increase in surface temperature since 1900.  
Hypothesis: We can predict a substantial fraction of the variability in surface temperature increase from pollution indices using a multi-layer convnet.    

Notice that we didn't start with the dataset and that is important. It may seem easier to start with a dataset + your favorite neural network, but that makes it very difficult to find a good question that matches that dataset. When you start with the question, you guarantee that you are personally invested in the question, and that you can have a unique angle or perspective that others may have not thought about yet.

To find the dataset, you can rely on searches, especially using the keywords you identified in your literature review. You can also go to a few very well organized websites/repositories that have links to datasets (see [these ones](https://deeplearning.neuromatch.io/projects/docs/datasets_and_models.html) for example). When you find a dataset, make sure you can load it easily into google colab, either by downloading it locally and then uploading to your google drive, OR (and this is preferable/easier, especially if you don't have good internet) by directly downloading the dataset into google colab. You should then start working on the reformatting of the data, to put it into a format that you can apply deep learning on it (usually a multi-dimensional numpy array, or a set of images with a dataloader object in Pytorch). It's good to organize your dataset into a nice format early on, because that will save you time later and will make it easier to think about how to apply models to your data.


## Project Day 6 (3h/day)

You should now have a question, a dataset and a hypothesis. One or more of these could be shaky/vague, and that is ok! As you start testing you hypothesis, the weak points of your project will become clear and then you can update what needs updating (question, dataset or hypothesis).

The next modelling steps are steps 5-6 in the jupyterbook materials: selecting a toolkit and planning a model. For deep learning, the "toolkit" could be the specific flavor of deep learning model you want to implement (transformer, residual network etc.). The model planning would be the specific architecture you give to your model (how many layers, convolution size etc). Design  your model in pytorch and try to get it to run. You don't need to start training it yet: first make sure it is designed correctly.

## Project Days 7-10 (3h/day)

These days correspond roughly to the modelling steps 7-9 videos and notebook, which primarily covers the implementation of the model and testing whether the model works correctly. Since you already have designed a neural network, it's time to try training it. This can take a lot of trial and error, but you don't need to wait for the entire training to finish. The cost function should decrease fairly quickly in the first few epochs, so try to optimize hyperparameters so that you get good initial reductions in the cost function. Warning: don't be too aggressive. The hyperparameters that work best for the first few iterations are likely to get stuck later on and/or lead to the parameters "exploding". It is often best to reduce the learning rate by a factor of 2 or so from the hyperparameters that lead to the best initial descent.

## Project Day 11 (8h, Monday)

This is a full project day, abstract writing day! One of the best ways to understand your own research is to try to write about it. You should write early and often, not just at the end when you’re trying to write a paper or your thesis. Science conferences are a great way to present your intermediate work, and they give you a chance to write an abstract. For example, the Neuromatch Conferences are a great venue for this. However, you don’t have to wait so long to write your project abstract, you’ll do it today! If you have preliminary results that’s great, but it’s not required. Most of the components of an abstract do not in fact require results. The goal for this day is to workshop your abstract with your group, and then present this to your entire pod. 

If you have been using a project template, this is a good time to branch out and pursue your own questions. The template was meant to get you started on some concrete analyses, so that you become familiar with the data, but now that you have more knowledge, you should be able to come up with your own question. Practice the first 4-steps again if necessary, they should be easier once you have a good question.

Your starting point for workshopping your abstract should be step 10 from the [Modeling steps 10](https://deeplearning.neuromatch.io/projects/modelingsteps/ModelingSteps_10_DL.html) notebook, and especially the example projects ([deep learning](https://deeplearning.neuromatch.io/projects/modelingsteps/Example_Deep_Learning_Project.html), [modelling](https://deeplearning.neuromatch.io/projects/modelingsteps/TrainIllusionModelingProjectDL.html) and [data science](https://deeplearning.neuromatch.io/projects/modelingsteps/TrainIllusionDataProjectDL.html)) which show how you can build an abstract if you have been following the 10 steps.

Note: the timings for this day are just suggestions. You can spend more or less time on different parts depending on how much work you think your abstract needs. Also, take as much time as you need in the beginning of this day to carefully go through the modelling steps notebooks (including the example projects).

With your group
* (30 min) use the ABC...G questions from the example deep learning / model / data projects to write your own answers and build a first version of your abstract.
* (30 min) by yourself, read the [Ten simple rules for structuring papers](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005619). Pay close attention to figure 1, and how it specifies the same overall structure for the abstract, for each paragraph in the paper, and for the paper as a whole.  
* (1h) workshop your abstract together as a group. Say what you like and what you don’t like about it. Try to refer back to the principles from the "Ten simple rules" paper in order to find problems with your abstract and follow the recommended solutions from the paper.   

1h break

* (30-60 min) Edit the abstract individually in your own google doc. At this stage, it is also important to control the flow of the abstract, in addition to keeping the structure from the 10 rules-paper. The flow relates to the “writing style”, which is generally no different for researchers than for other writers. Most importantly, make sure each sentence continues from where the previous one left, and do not use jargon without defining it first. Check out this book about writing if you have time ([book](https://sites.duke.edu/niou/files/2014/07/WilliamsJosephM1990StyleTowardClarityandGrace.pdf), especially chapter 3 about "cohesion" and flow.
* (30 min) You should now have as many copies of your abstract as there are students in your group. Put them all into the same google doc, and try to see what you all did the same / differently. What sounds better? Pick and choose different sentences from different abstracts.


With your project TA (timing is not precise!)
* (30-60min) Try to schedule a meeting with your project TA during this day and show them your abstract. Try to get explicit feedback and edit the abstract together in a google doc.

Last 3h, with the pod.

* (30min / group = 1h) It is always revealing to present your research to someone who has never heard about it. Take turns in your pod to read each other’s abstracts and provide feedback on them. Tell the other group what you understand and what you don’t from their research project. Give detailed writing feedback if you can (use "suggestion mode" in google docs). If there is no other project group in your pod, ask your TA to reach out to other pods to find a group you can workshop your abstract with.

Back in your group
* (1-2h) Has the abstract refined or changed your question? Use the rest of this day to make a concrete plan for the final week of your project. If you already answered your question, then you will need to plan for control analyses, maybe including some simulated data that you need to also generate yourself.

Once you are done, please submit the abstract [HERE](https://airtable.com/shrJZuCmrPRRjbYpE). We will use your submission to find a few other groups for your one hour mini-conference on Friday, during which you will present your project.

## Days 12-14 (3h/day)

Abstract writing day should have helped you narrow down what results (positive or negative) you would actually need to answer your question. You will use the rest of this time to try to get a result, or make progress towards an answer. This might not work out in such a short time, but don’t get discouraged: this part normally takes months if not years of work. You might need some new tools at this point to analyze your model, such as dimensionality reduction techniques, or visualization methods for units inside deep neural networks.

* If you know what analysis you need, but don’t know how to do it, the TAs are there to help you. They can point you to useful toolkits that may be difficult to find otherwise.
* Try not to implement complicated analyses from scratch. Use existing toolkits, and learn how to use them well. This kind of knowledge is very helpful long-term.
* If you find a negative answer to your question, that is absolutely ok! Please do report that. Then go back and think about how this affects your initial hypothesis. Does it rule it out, or could there be limitations in this particular data that lead to the negative result? What other data would you collect that would be better suited for answering this question? Try to design a new experiment in very specific detail and tell us about it. Who knows, somebody might run that experiment someday!
* If you find a positive result (i.e. the data matches your hypothesis), then you should spend the rest of your time validating it to make absolutely sure it is really true. You will need to design controls using the data (shuffling controls), or using simulated data, and you need to check the logic of your pipeline from start to end. Did you accidentally select only neurons that were tuned to a behavior, and then showed that they respond to aspects of that behavior? Did you sort neurons by their peak response time and then found sequences in your data? That is circular reasoning! There are some obvious and some not-so-obvious circular analyses that can catch even experienced researchers off-guard. This is what the controls are especially useful at catching.  

At the end of project day 14, you should also submit your slides via this [Airtable link]().

## Day 15 Presentations (last day!)

This is the day where you present your project to other groups who worked on the same topic. Your project TA will be invited too, but they are busy so they might not make it. The groups will take turns to share their screens. You can use figures and other graphics, but this is meant to be told as a story, and everyone from your group should take a turn telling a part of the story. Tell us about the different hypotheses you’ve had at different points and how you refined them using some of the tools we taught.

### Schedule

* 10 minutes of meet & greet. Do a round of introductions (one TA calls out names from the zoom list). Everyone says their name, pod name, position, university and subject of study, as well as one interesting fact about themselves "Hi, I'm Jonny from the wiggly caterpillars and I am a PhD student at University of Notre Dame in Paris. I build reinforcement learning models for robots, and in my free time I like to go on long bike rides".

* 30 minutes of presentations, including questions. Each group should speak for approx 5 minutes (depending on group size), and then take questions for 1-2 minutes. The order of presentations should be the one from the email.

* 20 minutes of discussion. Use the following questions to guide the group discussion. Spend 3-5 minutes on each question.  
  * Could one of the datasets chosen by the other groups have helped you answer your question, or a part of it? What does the other group think?
  * Does anyone plan to continue working on this project in the future? Perhaps a few students from the multiple groups would like to continue together?
  * Which one of the 10 steps to modelling/research was hardest and why?
  * Based on your experience with the NMA project, what project would you most like to do next? Make up your own, or pick from the NMA projects (a different dataset or project template which you did not use).
  * What surprised you the most about the process of doing a project? In what way was this project most different from other projects you have done in the past.
  * What technique did you learn at NMA which you think you can immediately apply to your own project (if you are currently doing research)?  


### Logistics:

* You will present to other groups (3-5 groups per room). An email will be sent with the zoom room of one of the pods where everyone goes for one hour corresponding to either:
  * timeslots 2,4:   last hour of project time, -1:00 to 0:00 relative to start of your normal tutorial time (check the [calendar](https://academy.neuromatch.io/calendar-summer-2021)).
  * timeslots 1,3,5: 6:10-7:10 relative to the start of your normal tutorial time (check the [calendar](https://academy.neuromatch.io/calendar-summer-2021)).
  There is a hard cutoff at the one hour mark, so the TAs must ensure everyone gets a turn to present.  

* Use this presentation style ([google slides](https://docs.google.com/presentation/d/1A1uaYarVot9YyCdbAAB4VDvsQfK6emqq-TwIZ9xVNwo/edit?usp=sharing) or [powerpoint](https://osf.io/ky6fj/download)) or create your own style!

* One minute per person and one slide per person only! This is primarily to ensure that everyone in your superpod gets to go before the hard cutoff at the one hour mark.

* Do not do introductions again, just present the material directly.

* When you are done presenting, leave the last slide up (with conclusions), and open the floor for questions.


### Content:

* The 1 minute, 1 slide rule might seem like an impossible limit. However, it is one of the most useful formats you can learn, often referred to as a "one minute elevator pitch". If you can become an expert at giving short pitches about your work, it will help you get the interest of a lot of people, for example when presenting posters at scientific conferences. Or when you accidentally find yourself in an elevator with Mark Zuckerberg: this could be your chance to secure a million dollars in research funds!

* The key to a good presentation is to practice it by yourself many times. It is no different from other performing arts (acting, playing a musical instrument etc), where rehearsals are so crucial to a good performance.

* If something in your presentation doesn't sound good or doesn't make sense, you WILL get annoyed by it when you say it the tenth time, and that will make you want to change it. (Secret: this how professors prepare all of their presentations and it's why they always sound like they know what they are talking about)

* Always have an introduction slide and a conclusion slide. If your group is relatively large (>=5 people), then someone should be assigned to each of the intro and conclusion slides. If your group is small, then the same person can give intro + next slide, or conclusion slide + previous slide.

* Short anecdotes can work like magic for engaging your audience. As a rule, most listeners are passive, bored, not paying attention. You have to grab their attention with that smart elevator pitch, or with a short anecdote about something that happened to your group while working on projects.

* Most groups won’t have a result and this is absolutely normal. However, the main goal anyway is to communicate the logic of your project proposal. Did you design a smart way to test the neural binding hypothesis, but then didn’t find the data to get answers? That can also be very interesting for others to hear about! Furthermore it will make it clear that research never stops. It continues as a series of questions and answers, not just within your own project, but at the level of the entire research field. Tell us what got you excited about this particular project, and try to dream big. One day, models like yours could be used to do what?


### Questions:

* If your presentation was short enough, there can be time for questions from the audience. These are a great way to get feedback on your project!

* Before you ask a question, consider whether others might be interested in that topic too. This usually means asking big picture questions, as opposed to detailed, technical questions, but there are exceptions.

* If you are answering the question, try to be short and concise. Rambling is very clear to the audience, and it can seem like you're avoiding to answer the question. Answering concisely is another very useful skill in "real life". It also means that you can take more questions given our time constraints.

<!--- Over the next two weeks, you will iterate between refining your question and trying to answer it. Be on the lookout for interesting hypotheses. You might notice something weird in the data, and if you dig deeper it might lead you directly to a result. For this to work, you must keep an open mind about what your questions are. If you feel like your question is starting to change, go back to Steps1-5 and see if it’s easier to formulate those steps with the new question. A good question/hypothesis makes the 5 steps really easy to think through. Here are some generally useful tips & tricks:

* The hardest part will be wrestling with the data to try to answer your question. You can rely on your TAs, the dedicated project TAs and the Discord channels to make this process easier.
* For theory projects, wrestling with your model can be equally challenging. If your model generates data, for example a neural network simulation, then you can still use some of the tricks below to analyze that data.
* If your model makes a hypothesis that needs to be tested, then your theory project might become a data project. The opposite may also happen: you may find something interesting in the data, and realize that you need a model to understand it better.  
* Always be on the lookout for bugs in your code, or ”bugs” in your analysis plan. If a plot/result looks too good to be true, it might be! Make sure you always split your data train/test, even for simple analyses where you think it might not matter (i.e. for making tuning curves).
* If your question does change, remember to always do a quick literature survey (i.e. google search) to see if others thought about your question in the past. You don’t need to come up with a completely original question! Do however situate your research within the relevant literature, and try to get hints/suggestions from other papers.
* Depending how complex your question is, there could be several data analysis steps:
   * data wrangling: some questions can be answered simply by plotting the right variable from the data! Some generally useful strategies: make PSTHs and tuning curves; try scatter plots of different variables; plot across neurons or across trials; select the most tuned neurons and look just at those; if there are multiple sessions pick a good one and dig deep into that one.
   * simple, linear analyses: most questions can be answered at this stage. This is often needed if you are doing a “population analysis”, i.e. trying to determine if a set of neurons or voxels collectively encode a certain variable. By far, the most used linear analyses are linear regression, PCA and k-means clustering.
   * Linear regression is often a good first step, even if your variables are binary/categorical. Once you have a pipeline, it will be easy to switch to logistic regression or other predictors from the scikit-learn library.
   * For visualizations, you might want to reduce a population of neurons to just a few components using PCA, then go back to the “data wrangling” steps and make the same kinds of plots for PCs that you made for neurons.
   * Another way to reduce the size of data is to cluster neurons (or trials!) into a few subsets, then average within that cluster. The simplest clustering model is k-means, which is a “linear” clustering model.
* complicated, nonlinear analyses: if the simple analyses fail, you might think you have no choice but to try something fancy, like deep learning or ISOMAP. This is often a dead end where projects go to die! You probably will make a lot more progress by slightly (or greatly) changing your question, or refining your hypothesis. The reason complicated analysis are so hard to do and interpret is that they often function as black boxes that are hard to look into. What do the parameters of a deep neural network mean? That is a hard research question in its own right. This is not to say that your hypothesis cannot be a nonlinear model, just that you can often test nonlinear hypotheses with simple, even linear analyses. If you must, however, use complicated analyses, then deep learning “replaces” linear regression, t-SNE / ISOMAP replaces PCA, and hdbscan replaces k-means.
   * deep learning as a prediction tool. This is unlikely to do better than linear/logistic regression, because neural data is noisy, and you need a lot of training data to really train a deep network well. This is because deep networks have a lot of parameters.
   * There are many “nonlinear dimensionality reduction” methods like t-SNE / ISOMAP, but these are often not meant as replacements for PCA, but instead as visualization tools to try to see a clustering structure in your data. They can be useful for making hypotheses based on interesting-looking plots. You still need to validate those hypotheses using simpler methods, like clustering and PCA.
   * There are many nonlinear clustering models like hdbscan and spectral clustering, but those are fickle for high-dimensional data and difficult to interpret. You will have to carefully try different parameters, and think through what the clusters mean.
