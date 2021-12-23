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
    // $('#inlineRadio1').click(function () {
    //   if ($('#inlineRadio1').val() != undefined) {
    //     $('#valNext').attr('disabled', false)
    //   }
    // })
    // $('#inlineRadio2').click(function () {
    //   if ($('#inlineRadio2').val() != undefined) {
    //     $('#valNext').attr('disabled', false)
    //   }
    // })
    // $('#inlineRadio3').click(function () {
    //   if ($('#inlineRadio3').val() != undefined) {
    //     $('#valNext').attr('disabled', false)
    //   }
    // })
    // $('#inlineRadio4').click(function () {
    //   if ($('#inlineRadio4').val() != undefined) {
    //     $('#valNext').attr('disabled', false)
    //   }
    // })
    // $('#inlineRadio5').click(function () {
    //   if ($('#inlineRadio5').val() != undefined) {
    //     $('#valNext').attr('disabled', false)
    //   }
    // })
    // $('#inlineRadio6').click(function () {
    //   if ($('#inlineRadio6').val() != undefined) {
    //     $('#valNext').attr('disabled', false)
    //   }
    // })
    // $('#inlineRadio7').click(function () {
    //   if ($('#inlineRadio7').val() != undefined) {
    //     $('#valNext').attr('disabled', false)
    //   }
    // })
  });