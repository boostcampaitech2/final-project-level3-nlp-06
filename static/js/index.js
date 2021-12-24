$(document).ready(function () {
    var i = 1;
    while ($('#corp_info'+i).length) {
        $('#corp_info'+i).hide();
        i += 1;
    }
    var j = 1;
    while ($('#keyword_graph'+j).length) {
        $('#keyword_graph'+j).hide();
        j += 1;
    }
    var k = 1;
    while ($('#inside_corp_info'+k).length) {
        $('#inside_corp_info'+k).hide();
        k += 1;
    }
});

function onDisplayCorporations(id) {
    var i = 1;
    while ($('#corp_info'+i).length) {
        if (i != id) {
            $('#corp_info'+i).hide();
        } else {
            if ($('#corp_info'+i).css('display') == 'none')
                $('#corp_info'+i).show();
            else
                $('#corp_info'+i).hide();
        }
        i += 1;
    }
}

function onDisplayGraphs(id) {
    var i = 1;
    while ($('#keyword_graph'+i).length) {
        if (i != id) {
            $('#keyword_graph'+i).hide();
        } else {
            if ($('#keyword_graph'+i).css('display') == 'none')
                $('#keyword_graph'+i).show();    
            else
                $('#keyword_graph'+i).hide();
                
        }
        i += 1;
    }
}

function onDisplayInsideCorporations(id) {
    var i = 1;
    while ($('#inside_corp_info'+i).length) {
        if (i != id) {
            $('#inside_corp_info'+i).hide();
        } else {
            if ($('#inside_corp_info'+i).css('display') == 'none')
                $('#inside_corp_info'+i).show();    
            else
                $('#inside_corp_info'+i).hide();
                
        }
        i += 1;
    }
}