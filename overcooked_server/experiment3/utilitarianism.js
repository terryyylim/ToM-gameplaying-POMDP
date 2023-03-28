/* initialize jsPsych */
var jsPsych = initJsPsych({
  on_finish: function() {
    //jsPsych.data.displayData();
    //var full = jsPsych.data.get();
    //console.log(full);
  }
});

/* create timeline */
var timeline = [];

// debriefing form
function debriefing_form() {
  var content = "<div style='text-align:left; width:700px; margin:0 auto'>"
  + "<h3>Great work. Finally, we just have a couple of questions for you!</h3>"
  + "<p>Did you read and understand the instructions correctly?<br><input required='true' type='radio' id='yes_understood' name='understand' value='yes'><label for='yes'>Yes</label><br><input required='true' type='radio' id='not_understood' name='understand' value='no'><label for='no'>No</label><br></p>"
  + "<p>Were there any problems or bugs in the study?<br><input required='true' name='problems' type='text' size='50' style='width:100%;border-radius:4px;padding:10px 10px;margin:8px 0;border:1px solid #ccc;font-size:15px'/></p>"
  + "<p>Age:<br><input required='true' name='age' type='number' style='width:20%;border-radius:4px;padding:10px 10px;margin:8px 0;border:1px solid #ccc;font-size:15px'/></p>"
  + "<p>Please indicate your gender:<br><input required='true' type='radio' id='male' name='gender' value='male'><label for='male'>Male</label><br><input required='true' type='radio' id='female' name='gender' value='female'><label for='female'>Female</label><br><input required='true' type='radio' id='other' name='gender' value='other'><label for='other'>Other</label></p>"
  + "<p>Any additional comments you would like to share?<br><input name='comments' type='text' size='50' style='width:100%;border-radius:4px;padding:10px 10px;margin:8px 0;border:1px solid #ccc;font-size:15px'/></p>";
  return content;
};

// generate a random subject ID with 8 characters
var subject_id = jsPsych.randomization.randomID(9);

jsPsych.data.addProperties({ID: subject_id});

//put all video files and image files into below
var videos = [
'video/maybe_line_new_1_1cut.mp4',
'video/maybe_line_new_2_1cut.mp4',
'video/maybe_line_new_3_1cut.mp4',
'video/maybe_line_new_4_1cut.mp4',
'video/maybe_line_new_5_1cut.mp4',
'video/maybe_line_new_6_1cut.mp4',
'video/no_line_1_1cut.mp4',
'video/no_line_2_1cut.mp4',
'video/no_line_3_1cut.mp4',
'video/yes_line_7_1cut.mp4',
'video/yes_line_8_1cut.mp4',
'video/yes_line_9_1cut.mp4',
'video/yes_line_10_1cut.mp4'
];

var all_videos = videos.concat('video/instruction_video.mp4')
var all_images = ['img/1.png','img/2.png','img/3.png','img/4.png','img/5.png','img/6.png','img/7.png','img/8.png'];

var matching_image = {
  "video/maybe_line_new_1_1cut.mp4": "img/8.png",
  "video/maybe_line_new_2_1cut.mp4": "img/8.png",
  "video/maybe_line_new_3_1cut.mp4": "img/2.png",
  "video/maybe_line_new_4_1cut.mp4": "img/8.png",
  "video/maybe_line_new_5_1cut.mp4": "img/6.png",
  "video/maybe_line_new_6_1cut.mp4": "img/5.png",
  "video/no_line_1_1cut.mp4": "img/6.png",
  "video/no_line_2_1cut.mp4": "img/8.png",
  "video/no_line_3_1cut.mp4": "img/8.png",
  "video/yes_line_7_1cut.mp4": "img/1.png",
  "video/yes_line_8_1cut.mp4": "img/1.png",
  "video/yes_line_9_1cut.mp4": "img/7.png",
  "video/yes_line_10_1cut.mp4": "img/2.png"
};

var preloadAll = {
  type: jsPsychPreload,
  images: all_images,
  video: all_videos
};
timeline.push(preloadAll);

/* define welcome message trial */
var welcome = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: '<p style="text-align:center;width:700px;"> Welcome to the experiment! This study takes most people around 15 minutes to complete. Please complete it in one sitting.</p>',
  choices: [" "],
  prompt: 'Press \'Spacebar\' to continue.'
};
timeline.push(welcome);

/* define consent form */
var consent = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: '<p style = "font-size:16px;">MIT Computational Cognitive Science Lab</p><h4>Press the \'y\' key to consent.</h4><div style = "background-color:#f2f2f2;"><h3>Informed Consent Form</h3><div style="text-align:left;width:700px;"><p style = "font-size:16px;">In this HIT, you will be asked to read a set of instructions that describe a particular scenario, and you will then be asked to perform a simple task relating to that scenario. There will also be comprehension and attention checks. Participants who do not answer these checks correctly cannot be approved. In order to get credit for this HIT, you must read all the questions fully and answer honestly. You may only complete this HIT once. If you complete this HIT multiple times, you will be rejected.</p><p style = "font-size:16px;">This HIT is part of a MIT scientific research project. Your decision to complete this HIT is voluntary. There is no way for us to identify you. The only information we will have, in addition to your responses, is the time at which you completed the survey. The results of the research may be presented at scientific meetings or published in scientific journals. Pressing the "Y" key indicates that you are at least 18 years of age, and agree to complete this HIT voluntarily. Press the "Y" key to confirm your agreement and continue.</p></div>',
  choices: ['y'],
};
timeline.push(consent);

/* define instructions trial */
var instructions1 = {
  type: jsPsychSurveyText,
  preamble: `
  <p>(Instructions 1 out of 3)</p>
  <p style="text-align:center;width:900px;">Please read the instructions very carefully. In this experiment, you will watch videos of a video game. Everyone in the game is trying to collect water from a water source and bring it back to the wooden storage buckets. Each person needs to collect one bucket of water for each level. The faster a person collects the bucket of water, the more points they receive. In each video, you will see just one person begin to leave the line. You will answer how much you think everyone is slowed down by this person leaving the line to go directly to the water themselves.</p>
  <p style="text-align:center;width:900px;">Answer the attention check question and press continue</p>
  `,
  questions:[{prompt: 'What is the goal of the people in the game?', required: true, name: 'instructions1'}],
  data:{trial_category: "instructions"}
};
timeline.push(instructions1);

var instructions2 = {
  type: jsPsychSurveyText,
  preamble: `
  <p>(Instructions 2 out of 3)</p>
  <p style="text-align:center;width:900px;">Please watch the video below to get a sense of what the environment is like, and what actions people can take. In the example video below, there is only 1 active player moving so you can focus on what it looks like when people collect water! But when the actual experiment begins, all the videos you see will have 8 people moving in them.</p>
  <p><video autoplay style="height:auto;width:600px;" src='video/instruction_video.mp4'></video></p>
  <p style="text-align:center;width:900px;">As you see in the video above, each person can move one square per time step in one of the four cardinal directions (up, down, left, right). They cannot move through squares with objects in them (like rocks, plants, wells, and water). They can collect water if they are in a square that is adjacent to a square with water, and it takes one time step to collect the water.</p>
  <p style="text-align:center;width:900px;">Something that you’ll notice when you see videos with more people in them, is that people can only move to squares that are not occupied by other people. In the game, the level ends when all 8 people collect water and store it in a wooden storage bucket.</p>
  <p style="text-align:center;width:900px;">Answer the attention check question and press continue</p>
  `,
  questions:[{prompt: 'How many squares can players move in each time step?', required: true, name: 'instructions2'}],
  data:{trial_category: "instructions"}
};
timeline.push(instructions2);

var instructions3 = {
  type: jsPsychSurveyText,
  preamble: `
  <p>(Instructions 3 out of 3)</p>
  <p style="text-align:center;width:900px;"> In each video, all 8 people will start out waiting in a line in front of one of the water sources. People can leave the line and head straight for the water source without waiting, or they can stay waiting in line until it’s their turn to collect water. The videos you see will end shortly after just one character makes a decision to leave the line. Your job is to think about how much everyone will be slowed down as a result of their decision.</p>
  <p style="text-align:center;width:900px;"> Feel free to watch the video many times, but please watch it at least once completely and think carefully before you make your judgments. Answer the attention check question and press continue to start the experiment!</p> 
  <p style="text-align:center;width:900px;"> After the experiment is over, you will have an opportunity to give us general feedback and let us know if anything was confusing or unclear.</p>
  `,
  questions:[{prompt: 'What are you answering for each video?', required: true, name: 'instructions3'}],
  data:{trial_category: "instructions"}
};
timeline.push(instructions3);

//begin experiment trials

/* Randomize array in-place using Durstenfeld shuffle algorithm */
function shuffleArray(array) {
  for (var i = array.length - 1; i > 0; i--) {
    var j = Math.floor(Math.random() * (i + 1));
    var temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }
}

shuffleArray(videos);

for (var i = 0; i < videos.length; i++){
  var stim_name = videos[i]
  var stim_category = ""

  if (stim_name.includes("no")){
    stim_category = "no_1"
  }
  else if (stim_name.includes("maybe")){
    stim_category = "maybe_1"
  }
  else{
    stim_category = "yes_1"
  }

  var video_trial = {
    type: jsPsychSurveyText,
    preamble:'<p><video controls autoplay style="height:auto;width:600px;" src=' + videos[i] + '></video></p>',
    questions: [{prompt: '<p><img src=' + matching_image[videos[i]] + '></img></p> <p style="text-align:center;width:700px;">How much slower or faster will it be for everyone given this character is leaving the line? State a number between -50 to 50, where -50 means much slower for everyone, 0 means about the same time, and 50 means much faster for everyone. Please explain your decision.</p>', required: true, rows: 3}],
    data: {
      trial_category: stim_category,
      stimulus_name: videos[i]
    }
  }
  timeline.push(video_trial);


  // var video_trial = {
  //   type: jsPsychVideoSliderResponse,
  //   stimulus: [videos[i]],
  //   labels: ['Much slower', 'The same amount of time', 'Much faster'],
  //   response_allowed_while_playing: false,
  //   min: -50,
  //   max: 50,
  //   slider_start: 0,
  //   prompt: '<p><img src=' + matching_image[videos[i]] + '></img></p> <p style="text-align:center;width:700px;">How much slower or faster will it be for everyone given this character is leaving the line? Slide the bar to indicate your answer.</p>',
  //   data: {
  //     trial_category: stim_category,
  //     stimulus_name: videos[i]
  //   },
  //   rate: 1.5,
  //   slider_width: 500,
  //   controls:true,
  //   width: 600
  // }
  // timeline.push(video_trial);
}

/* define debrief */
var debrief_qs = {
  type: jsPsychSurveyHtmlForm,
  html: debriefing_form(),
  data: {trial_category: 'debrief'},
  dataAsArray: true,
  on_finish: function(data){}
};
timeline.push(debrief_qs);

var conclusion = {
  type: jsPsychSurveyText,
  preamble: '<div style="font-size:20px;">This task is now over. The completion code is: ' + subject_id + ' Thank you for your participation! </div>',
  questions: [{prompt: 'confirm by entering your code here, and press continue', required: true}]
};
timeline.push(conclusion)

/* start the experiment */
jsPsych.run(timeline);