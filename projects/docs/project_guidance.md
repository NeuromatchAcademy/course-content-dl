# Daily guide for projects

## Summary

This project plan explicitly encourages the iterative nature of research as a series of questions and answers that gradually refine your hypotheses. We have assigned you to pods based on your broad research interests (neuroscience, computer vision, natural language processing or reinforcement learning). Each pod will split into two groups alphabetically. First sort yourselves by the first letter of your name. The first half of the students are in group 1, the second in group 2.

Once you're in groups, you will start brainstorming and via searching the literature for interesting papers, with the goal of forming a project question. For the rest of week 1 you will look for an appropriate dataset to answer your question, and try to process that dataset into a format that is good for modeling. Week 1 day 4 (W1D4) is entirely dedicated to projects.

In the second week you will experiment with different types of deep neural networks to try to answer your question. By W3D2, you will be able to write a short abstract about your project, which may or may not include results, but it should at least include a testable hypothesis. For the rest of the week, you will focus on getting evidence for/against your hypothesis.

Finally, on the last day of the course (W3D5), you will meet with other groups and tell them the story of your project. This is a low-key presentation that may include some of the plots you made along the way, but it is not meant as a real research presentation with high “production values”. See some of the [examples](https://compneuro.neuromatch.io/projects/docs/project_2020_highlights.html) from the comp-neuro course last year to get a sense of what the presentation should look like.

## Project templates

[Project templates](https://deeplearning.neuromatch.io/projects/docs/projects_overview.html) are research ideas developed by the NMA team or by the project TAs which contain code and slides with background and questions. Project templates can be used in a variety of ways.
* For starters, you can use the project templates just to get familiarized with some neural network applications. They can provide keywords for you to use in your literature search, or python libraries you can reuse to answer your own questions.
* You may start with a project template, use it in the first week and then in the second week diverge from it as your group develops their own new idea or question to test.
* Templates have a natural flow of questions, but don’t hesitate to skip or completely change some of these. They are meant to be used very flexibly!

## Project TAs

Project TAs are your friendly topic experts to consult with on all issues related to your project. They can help with brainstorming, literature searches and coding. You will have one-hour meetings with an assigned project TAs starting Tuesday and continuing every 2-3 days until the end of the school. During this time, they will help you refine your question and hypothesis into something that can be answered with deep learning. Sometimes they may miss a meeting or arrive late (busy schedules, lots of meetings!). In those cases, please stop what you were doing to have the meeting, and then resume your work when the project TA leaves.

Sometimes the project TAs might need to schedule meetings slightly earlier or later. Please try to be flexible, they are doing a lot of "virtual footwork" to make sure all the groups have enough support during projects. We also encourage you to reach out to them for extra meetings whenever you need them, and to post questions on discord in the #topic channels. All project TAs have time set aside specifically to answer discord questions and to provide additional meetings when necessary.


## Week 1: Getting started

Depending on your time slot, you may or may not have project time on the first day of the course. Regardless of whether your first project meeting is day 1 or day 2, spend your first session doing the following:

* Split into groups alphabetically, as described above.
* Introductions (30 min = 2 min/student): say a few things about yourself, then about your research area or research interests. What are you really curious about, that you might explore in your NMA project?
* Listen carefully as others talk about their interests.
* Individual reading time (30 min): browse the projects booklet which includes this guide (skim the entire thing) + 16 project templates with slides and code + docs with further ideas and datasets
* Brainstorming (60 min): do you have an idea or a rough direction? Bring it out to the group.
* Now brainstorm within your group (60 min). Definitely choose a topic and start thinking about concrete questions if you can. Make sure the topic you choose is well suited for answering the broad range of questions you are interested in. Try to come up with one or a few topics of interest, either by yourselves or directly from the booklet (i.e. project templates).

Tips:
* No need to have a very concrete project after this day. You will determine the feasibility of your questions in the next few days, and you will likely change your question completely. That’s how research works!

In your next sessions, watch the [Modeling Steps 1-2 tutorials](https://deeplearning.neuromatch.io/projects/modelingsteps/ModelingSteps_1through2_DL.html). Continue brainstorming ideas. In practice, brainstorming and looking through related work are intertwined. You might want to get a headstart on looking at some literature, because it can inform and change your question.

* You will need to use your own project for some of this content. If you don’t yet have concrete ideas, or you haven’t done a research project before, use one of the provided project templates.
* If you are using a project template, your goal is to translate the information from the slide and colab notebook into the 10-steps format. Some information might not be readily available in the slide or notebook, and you might have to find it in your literature review later this day.
* Try to write down a few sentences for each of the two steps applied to your project. Putting thoughts into well-defined sentences and paragraphs helps at all stages of a project.

## W1D4: Projects Day!

This is a full day dedicated to projects! The goals are threefold: perform a literature search, refine your question, and try to find a good dataset.

Start by watching the videos for [Modeling Steps 3-4](https://deeplearning.neuromatch.io/projects/modelingsteps/ModelingSteps_3through4_DL.html). It will help you define the ingredients for your project. In our case, this primarily involves finding a dataset. Finding a good dataset is a crucial step in all deep learning projects. It will make the rest of the project much easier, so you should spend a lot of time on this (both days). This goes hand in hand with formulating a hypothesis (step 4), because some datasets will naturally invite you to make a hypothesis, especially if you already have a well-defined question. Here's an example:

Question:  What is the relation between pollution and climate change?
Dataset:   1) Geographic dataset of pollution indices; 2) Geographic dataset of average increase in surface temperature since 1900.
Hypothesis: We can predict a substantial fraction of the variability in surface temperature increase from pollution indices using a multi-layer convnet.
Notice that we didn't start with the dataset and that is important. It may seem easier to start with a dataset + your favorite neural network, but that makes it very difficult to find a good question that matches that dataset. When you start with the question, you guarantee that you are personally invested in the question, and that you can have a unique angle or perspective that others may have not thought about yet.

To find the dataset, you can rely on searches, especially using the keywords you identified in your literature review. You can also go to a few very well organized websites/repositories that have links to datasets (see [these ones](https://deeplearning.neuromatch.io/projects/docs/datasets_and_models.html) for example). When you find a dataset, make sure you can load it easily into google colab, either by downloading it locally and then uploading to your google drive, OR (and this is preferable/easier, especially if you don't have good internet) by directly downloading the dataset into google colab. You should then start working on the reformatting of the data, to put it into a format that you can apply deep learning on it (usually a multi-dimensional numpy array, or a set of images with a dataloader object in Pytorch). It's good to organize your dataset into a nice format early on, because that will save you time later and will make it easier to think about how to apply models to your data.


## W1D5 and W2D1 (3h/day): Toolkits and models

You should now have a question, a dataset and a hypothesis. One or more of these could be shaky/vague, and that is ok! As you start testing you hypothesis, the weak points of your project will become clear and then you can update what needs updating (question, dataset or hypothesis).

Continue to [Modeling Steps 5-6](https://deeplearning.neuromatch.io/projects/modelingsteps/ModelingSteps_5through6_DL.html), selecting a toolkit and planning a model. For deep learning, the "toolkit" could be the specific flavor of deep learning model you want to implement (transformer, residual network etc.). The model planning would be the specific architecture you give to your model (how many layers, convolution size etc). Design  your model in pytorch and try to get it to run. You don't need to start training it yet: first make sure it is designed correctly.

## W2D2 to W3D1 (3h/day): Implementation

These days correspond roughly [Modeling Steps 7-9](https://deeplearning.neuromatch.io/projects/modelingsteps/ModelingSteps_7through9_DL.html), which primarily covers the implementation of the model and testing whether the model works correctly. Since you already have designed a neural network, it's time to try training it. This can take a lot of trial and error, but you don't need to wait for the entire training to finish. The cost function should decrease fairly quickly in the first few epochs, so try to optimize hyperparameters so that you get good initial reductions in the cost function. Warning: don't be too aggressive. The hyperparameters that work best for the first few iterations are likely to get stuck later on and/or lead to the parameters "exploding". It is often best to reduce the learning rate by a factor of 2 or so from the hyperparameters that lead to the best initial descent.

## W3D2: Half project day!

Half of this day is dedicated to projects, specifically to writing your abstract. One of the best ways to understand your own research is to try to write about it. You should write early and often, not just at the end when you’re trying to write a paper or your thesis. Science conferences are a great way to present your intermediate work, and they give you a chance to write an abstract. For example, the Neuromatch Conferences are a great venue for this. However, you don’t have to wait so long to write your project abstract, you’ll do it today! If you have preliminary results that’s great, but it’s not required. Most of the components of an abstract do not in fact require results. The goal for this day is to workshop your abstract with your group, and then present this to your entire pod.

If you have been using a project template, this is a good time to branch out and pursue your own questions. The template was meant to get you started on some concrete analyses, so that you become familiar with the data, but now that you have more knowledge, you should be able to come up with your own question. Practice the first 4-steps again if necessary, they should be easier once you have a good question.

Your starting point for workshopping your abstract should be step 10 from the [Modeling steps 10](https://deeplearning.neuromatch.io/projects/modelingsteps/ModelingSteps_10_DL.html) notebook, and especially the example projects ([deep learning](https://deeplearning.neuromatch.io/projects/modelingsteps/Example_Deep_Learning_Project.html), [modeling](https://deeplearning.neuromatch.io/projects/modelingsteps/TrainIllusionModelingProjectDL.html) and [data science](https://deeplearning.neuromatch.io/projects/modelingsteps/TrainIllusionDataProjectDL.html)) which show how you can build an abstract if you have been following the 10 steps.

Note: the timings for this day are just suggestions. You can spend more or less time on different parts depending on how much work you think your abstract needs. Also, take as much time as you need in the beginning of this day to carefully go through the modelling steps notebooks (including the example projects).

With your group
* use the example deep learning / model / data projects to write your own answers and build a first version of your abstract.
* by yourself, read the [Ten simple rules for structuring papers](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005619). Pay close attention to figure 1, and how it specifies the same overall structure for the abstract, for each paragraph in the paper, and for the paper as a whole.
* workshop your abstract together as a group. Say what you like and what you don’t like about it. Try to refer back to the principles from the "Ten simple rules" paper in order to find problems with your abstract and follow the recommended solutions from the paper.

Then,
* Edit the abstract individually in your own google doc. At this stage, it is also important to control the flow of the abstract, in addition to keeping the structure from the 10 rules-paper. The flow relates to the “writing style”, which is generally no different for researchers than for other writers. Most importantly, make sure each sentence continues from where the previous one left, and do not use jargon without defining it first. Check out this book about writing if you have time ([book](https://sites.duke.edu/niou/files/2014/07/WilliamsJosephM1990StyleTowardClarityandGrace.pdf), especially chapter 3 about "cohesion" and flow.
* You should now have as many copies of your abstract as there are students in your group. Put them all into the same google doc, and try to see what you all did the same / differently. What sounds better? Pick and choose different sentences from different abstracts.


Try to schedule a meeting with your project TA and/or mentor during this day and show them your abstract. Try to get explicit feedback and edit the abstract together in a google doc.

Likewise, it is always revealing to present your research to someone who has never heard about it. Take turns in your pod to read the other group's abstract and provide feedback. What did you understand and what didn't make sense? Give detailed writing feedback if you can (use "suggestion mode" in google docs). If there is no other project group in your pod, ask your TA to reach out to other pods to find a group you can workshop your abstract with.

Finally, with your group, has the abstract refined or changed your question? Use the rest of this day to make a concrete plan for the final week of your project. If you already answered your question, then you will need to plan for control analyses, maybe including some simulated data that you need to also generate yourself.

Once you are done, please submit the abstract [here](https://airtable.com/shrUeDqzGe8Cplk8u).

## W3D3 and W3D4 (3h/day): Results

Abstract writing day should have helped you narrow down what results (positive or negative) you would actually need to answer your question. You will use the rest of this time to try to get a result, or make progress towards an answer. This might not work out in such a short time, but don’t get discouraged: this part normally takes months if not years of work. You might need some new tools at this point to analyze your model, such as dimensionality reduction techniques, or visualization methods for units inside deep neural networks.

* If you know what analysis you need, but don’t know how to do it, the TAs are there to help you. They can point you to useful toolkits that may be difficult to find otherwise.
* Try not to implement complicated analyses from scratch. Use existing toolkits, and learn how to use them well. This kind of knowledge is very helpful long-term.
* If you find a negative answer to your question, that is absolutely ok! Please do report that. Then go back and think about how this affects your initial hypothesis. Does it rule it out, or could there be limitations in this particular data that lead to the negative result? What other data would you collect that would be better suited for answering this question? Try to design a new experiment in very specific detail and tell us about it. Who knows, somebody might run that experiment someday!
* If you find a positive result (i.e. the data matches your hypothesis), then you should spend the rest of your time validating it to make absolutely sure it is really true. You will need to design controls using the data (shuffling controls), or using simulated data, and you need to check the logic of your pipeline from start to end. Did you accidentally select only neurons that were tuned to a behavior, and then showed that they respond to aspects of that behavior? Did you sort neurons by their peak response time and then found sequences in your data? That is circular reasoning! There are some obvious and some not-so-obvious circular analyses that can catch even experienced researchers off-guard. This is what the controls are especially useful at catching.

At the end of W3D4, you should also submit your slides via this [Airtable link](https://airtable.com/shr5NJa397fSYNDsO).

## W3D5 (last day!): Presentations

This is the day where you present your project to other groups. You can invite your project TA and mentor, but they are busy so they might not make it. The groups will take turns to share their screens. You can use figures and other graphics, but this is meant to be told as a story, and everyone from your group should take a turn telling a part of the story. Tell us about the different hypotheses you’ve had at different points and how you refined them using some of the tools we taught.

### Schedule

* 10 minutes of meet & greet. Do a round of introductions (one TA calls out names from the zoom list). Everyone says their name, pod name, position, university and subject of study, as well as one interesting fact about themselves "Hi, I'm Jonny from the wiggly caterpillars and I am a PhD student at University of Notre Dame in Paris. I build reinforcement learning models for robots, and in my free time I like to go on long bike rides".

* 30-40 minutes of presentations, including questions. Each group should speak for approx 5 minutes (depending on group size), and then take questions for 1-2 minutes. The order of presentations should be the one from the email.

* 10-20 minutes of group discussion. Use the following questions to guide the group discussion. Spend a few minutes on one or two of the questions below, or ask your own question.
  * Could one of the datasets chosen by the other groups have helped you answer your question, or a part of it? What does the other group think?
  * Does anyone plan to continue working on this project in the future? Perhaps a few students from the multiple groups would like to continue together?
  * Which one of the 10 steps to modelling/research was hardest and why?
  * Based on your experience with the NMA project, what project would you most like to do next? Make up your own, or pick from the NMA projects (a different dataset or project template which you did not use).
  * What surprised you the most about the process of doing a project? In what way was this project most different from other projects you have done in the past.
  * What technique did you learn at NMA which you think you can immediately apply to your own project (if you are currently doing research)?


### Logistics:
* Check the general schedule for precise timing of events on W3D5.
* You will present to other groups (3-5 groups per room).
*  There is a strict schedule, so the TAs must ensure everyone gets a turn to present.

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
