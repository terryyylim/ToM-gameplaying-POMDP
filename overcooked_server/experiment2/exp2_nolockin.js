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
'video/no_line_1_1cut.mp4', 'video/no_line_1_0cut.mp4',
'video/yes_line_7_1cut.mp4', 'video/yes_line_7_0cut.mp4'];

var all_videos = videos.concat('video/instruction_video.mp4')
var all_images = ['img/1.png','img/2.png','img/3.png','img/4.png','img/5.png','img/6.png','img/7.png','img/8.png'];

var matching_image = {
  "video/no_line_1_0cut.mp4": "img/2.png",
  "video/no_line_1_1cut.mp4": "img/6.png",
  "video/yes_line_7_0cut.mp4": "img/2.png",
  "video/yes_line_7_1cut.mp4": "img/1.png"
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
  stimulus: '<p style="text-align:center;width:700px;"> Welcome to the experiment! This study takes most people around 10 minutes to complete. Please complete it in one sitting.</p>',
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
  <p>(Instructions 1 out of 4)</p>
  <p style="text-align:center;width:900px;">In this experiment, you will watch videos of a video game. Everyone in the game is trying to collect water from a water source and bring it back to the wooden storage buckets. Each person needs to collect one bucket of water for each level. The faster a person collects the bucket of water, the more points they receive. In each video, you will focus on one person. When the video is over, you will answer whether you believe that person's actions were morally acceptable or morally unacceptable. </p>
  <p style="text-align:center;width:900px;">Answer the attention check question and press continue</p>
  `,
  questions:[{prompt: 'What is the goal of the people in the game?', required: true, name: 'instructions1'}],
  data:{trial_category: "instructions"}
};
timeline.push(instructions1);

var instructions2 = {
  type: jsPsychSurveyText,
  preamble: `
  <p>(Instructions 2 out of 4)</p>
  <p style="text-align:center;width:900px;">Please watch the video below to get a sense of what the environment is like, and what actions people can take. In the example video below, there is only 1 active player moving so you can focus on what it looks like when people collect water! But when the actual experiment begins, all the videos you see will have 8 people moving in them.</p>
  <p><video autoplay loop style="height:auto;width:600px;" src='video/instruction_video.mp4'></video></p>
  <p style="text-align:center;width:900px;">As you see in the video above, each person can move one square per time step in one of the four cardinal directions (up, down, left, right). They cannot move through squares with objects in them (like rocks, plants, wells, and water). They can collect water if they are in a square that is adjacent to a square with water, and it takes one time step to collect the water.</p>
  <p style="text-align:center;width:900px;">Something that you’ll notice when you see videos with more people in them, is that people can only move to squares that are not occupied by other people. Finally, each video will end when all 8 people collect water and store it in a wooden storage bucket.</p>
  <p style="text-align:center;width:900px;">Answer the attention check question and press continue</p>
  `,
  questions:[{prompt: 'When will each video end?', required: true, name: 'instructions2'}],
  data:{trial_category: "instructions"}
};
timeline.push(instructions2);

var instructions3 = {
  type: jsPsychSurveyText,
  preamble: `
  <p>(Instructions 3 out of 3)</p>
  <p style="text-align:center;width:900px;">In each video, all 8 people start out waiting in a line. Sometimes people will leave the line to try to get water more quickly for themselves. With each video, you will be shown an image of the specific person who you should focus on. This is the person whose actions you will judge as morally acceptable or morally unacceptable. Feel free to watch the video many times, but please watch it at 
    least once completely before you make your judgments!</p>
  <p style="text-align:center;width:900px;"> After the experiment is over, you will have an opportunity to give us general feedback and let us know if anything was confusing or unclear.</p>
  <p style="text-align:center;width:900px;"> Answer the attention check question and press continue to start the experiment!</p> 
  `,
  questions:[{prompt: 'How many people will you focus on to judge in each video?', required: true, name: 'instructions3'}],
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
    if (stim_name.includes("no_line_1_0cut")){
      stim_category = "no_0"
    }
    else {
      stim_category = "no_1"
    }
  }
  else{
    if (stim_name.includes("0cut")){
      stim_category = "yes_0"
    }
    else {
      stim_category = "yes_1"
    }
  }

 var video_trial = {
  type: jsPsychVideoButtonResponse,
  stimulus: [videos[i]],
  choices: ['morally unacceptable', 'morally acceptable'],
  response_allowed_while_playing: false,
  prompt: '<p style="text-align:center;width:700px;">Judge the behavior of this player in the video.</p> <p><img src=' + matching_image[videos[i]] + '></img></p>',
  data: {
    trial_category: stim_category,
    stimulus_name: videos[i]
  },
  rate: 1,
  controls:true,
  width: 600
  }
  timeline.push(video_trial);
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
  type: jsPsychHtmlKeyboardResponse,
  stimulus: '<div style="font-size:20px;">This task is now over. The completion code is: ' + subject_id + ' Press space after you have the completion code to finish. Thank you for your participation! </div>',
  choices: [' ']
};
timeline.push(conclusion)

/* start the experiment */
jsPsych.run(timeline);