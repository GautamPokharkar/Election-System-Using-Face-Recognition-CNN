 // static/js/main.js
document.addEventListener('DOMContentLoaded', function(){
  var tooltipTriggerList = [].slice.call(document.querySelectorAll('[title]'));
  tooltipTriggerList.forEach(function (el) {
    el.addEventListener('mouseenter', function(){
      // basic hover helper: show native title - browsers do this automatically
    });
  });
});

