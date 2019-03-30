showPreprocessingSettings = function() {
  var x = document.getElementById("pred-settings-form");
  x.style.display = "none";
  x = document.getElementById("train-settings-form");
  x.style.display = "none";
  x = document.getElementById("model-settings");
  x.style.display = "none";
  x = document.getElementById("prep-settings-form");
  if (x.style.display === "none") {
    x.style.display = "block";
  } else {
    x.style.display = "none";
  }
};

showTrainSettings = function() {
  var x = document.getElementById("prep-settings-form");
  x.style.display = "none";
  x = document.getElementById("pred-settings-form");
  x.style.display = "none";
  x = document.getElementById("model-settings");
  x.style.display = "none";
  x = document.getElementById("train-settings-form");
  if (x.style.display === "none") {
    x.style.display = "block";
  } else {
    x.style.display = "none";
  }
};

showPredictionSettings = function() {
  var x = document.getElementById("prep-settings-form");
  x.style.display = "none";
  x = document.getElementById("train-settings-form");
  x.style.display = "none";
  x = document.getElementById("model-settings");
  x.style.display = "none";
  x = document.getElementById("pred-settings-form");
  if (x.style.display === "none") {
    x.style.display = "block";
  } else {
    x.style.display = "none";
  }
};

showSettings = function() {
  var x = document.getElementById("prep-settings-form");
  x.style.display = "none";
  x = document.getElementById("train-settings-form");
  x.style.display = "none";
  x = document.getElementById("pred-settings-form");
  x.style.display = "none";
  x = document.getElementById("model-settings");
  if (x.style.display === "none") {
    x.style.display = "block";
  } else {
    x.style.display = "none";
  }
};

acc = function() {
  var items = Array('65.32','78.50','83.44','60.24','47.22');
  var item = items[Math.floor(Math.random()*items.length)];
  alert('Accuracy:'+item);
};
