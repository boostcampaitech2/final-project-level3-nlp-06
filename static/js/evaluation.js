$(document).ready(function() {
  $('#nickname').on('input', function() {
    if ($('#nickname').val() == '')
      $('#start_test').attr('disabled', true)
    else
      $('#start_test').attr('disabled', false)
  })
  $('#eval1').click(function () {
    if ($('#eval1').val() != undefined) {
      $('#valNext').attr('disabled', false)
    }
  })
  $('#eval2').click(function () {
    if ($('#eval2').val() != undefined) {
      $('#valNext').attr('disabled', false)
    }
  })
  $('#eval3').click(function () {
    if ($('#eval3').val() != undefined) {
      $('#valNext').attr('disabled', false)
    }
  })
  $('#eval4').click(function () {
    if ($('#eval4').val() != undefined) {
      $('#valNext').attr('disabled', false)
    }
  })
  $('#eval5').click(function () {
    if ($('#eval5').val() != undefined) {
      $('#valNext').attr('disabled', false)
    }
  })
  $('#eval6').click(function () {
    if ($('#eval6').val() != undefined) {
      $('#valNext').attr('disabled', false)
    }
  })
  $('#eval7').click(function () {
    if ($('#eval7').val() != undefined) {
      $('#valNext').attr('disabled', false)
    }
  })
});