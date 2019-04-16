// global variables
var pageLength = 5;
var response = null;
var urls = null;
var docTitles = null;
var page = 0;

// html generating funtions

// create html for list view
function htmlListView( entries, weights ) {

	var html = '';

	for(var i = page*pageLength; i < Math.min((page+1)*pageLength, entries.length); i++) {
		html += htmlEntry(entries[i], weights) + '\n\n'
	}

	return html;

}

// create html for status
function htmlStatus( metadata ) {

	html = '<span class="status">';

	if ( page==0 ) {
		html += 'Found ';
	} else {
		html += 'Page ' + (page+1) + ' of ';
	}

    numDocs = urls.length;
    numArgs = response.sentences.length;
    numPro = $.grep(response.sentences, function( n, i ) { return n.stanceLabel=='pro'} ).length;
	numContra = $.grep(response.sentences, function( n, i ) { return n.stanceLabel=='contra'} ).length;


	html += numArgs + ' arguments (' + numPro +' pro; ' + numContra +' con) in ';
	html += numDocs +' documents (classified ' + metadata.totalClassifiedSentences + ' sentences in ' + metadata.timeTotal.toFixed(3) +' ms)';

	html += '</span>';

	return html;
}

// generate html code for page navigation
function htmlPaging( numPages ) {

	html = '<div class="paging">';

    // 'prev' page
    if (page > 0) {
        html += '<span id="prev">Prev</span>';
    }

    var numLinks = 7;

    var start =  Math.max( Math.floor(page - numLinks/2), 0);
    var end = Math.min(start + numLinks, numPages);


    if((start > numPages - numLinks) && (numPages - numLinks >= 0)) {
        start = numPages - numLinks
    }

    for (var i = start; i < end; i++){
    	if (i == page){
    		html += '<span class="currentpage">' + (i + 1).toString() + '</span>'
    	}else{
    		html += '<span class="page">' + (i + 1).toString() + '</span>'
    	}
    }

    // 'next' page
    if (page != numPages - 1){
		html += '<span id="next">Next</span>'
	}

    html += '</div>'

    return html
}

// create html for procon view
function htmlProConView( sentences, weights ) {

	var pro = $.grep(sentences, function( n, i ) { return n.stanceLabel=='pro'} );
	var contra = $.grep(sentences, function( n, i ) { return n.stanceLabel=='contra'} );

	var html = '';

	html = '<table><tr><td width="50%" valign="top">';
    html += htmlListView( pro, weights );
    html += '</td><td width="50%" valign="top">';
    html += htmlListView( contra, weights );
    html += '</td></tr></table>';

	return html;
}

// build html for a single result entry
function htmlEntry( entry, weights ) {
	var html = '';

	// check stance
	if(entry.stanceLabel == 'pro') {
		html += '<div class="result"><span class="pro">PRO:</span>';
	} else {
		html += '<div class="result"><span class="con">CON:</span>';
	}

	if ( weights ) {
       	// process alphas
       	words = entry['sentencePreprocessed'].replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;").split(' ');
        alpha = entry['weights'];

       	delta = Math.max(...alpha) - Math.min(...alpha);

    	if (delta == 0) {
        	delta = 0.000001
    	}

    	// set hsl values depending on stance
    	if (entry.stanceLabel == 'pro') {
    		m = -0.62 / delta;
	    	b = 1.0 - m * Math.min(...alpha);
	    	hue = 85;
	    	saturation = 80;
		} else {
			m = -0.6 / delta
	    	b = 1.0 - m * Math.min(...alpha)
	    	hue = 0;
	    	saturation = 68;
		}

        for( var i = 0; i < words.length; i++){
        	word = words[i];

        	// capitalise first word of the sentence
        	if(i==0){
        		word = word.substring(0,1).toUpperCase() + word.substring(1)
        	}

        	// calculate luminance
        	luminance = (alpha[i] * m + b) * 100;

            html += '<span style="border-bottom: 2px solid hsl(' + hue + ','+ saturation +'%,' + luminance + '%) ;">' + word + '</span> ';
        }

    } else {
    	html += entry.sentenceOriginal.replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;");
    }

	var domain = entry.url.split('/')[2].replace('www.','');

	confidence = ((entry.argumentConfidence + entry.stanceConfidence) / 2).toFixed(4);
	html += '<span class="conf_score">(' + confidence + ')</span>\n';
	var date = new Date(entry.date)
	html += '<div class="url"><a target="_blank" href="' + entry.url + '"> Source: ' + domain + '</a>'
	html += '<span class="date">(' + date.toLocaleDateString('de-DE', {year: 'numeric', month: 'short', day: 'numeric'}) + ')</span></div></div>';

	return html;
}

// search for query input
function search() {
	query = document.getElementById('queryInput').value;
	sessionStorage.removeItem("queryResponse");

	if (!isAlphaNumeric(query)) {
		alert("The input query must only contain alphanumerical symbols.")
	}else{
		$("#resultBox").css('display', 'none');
		$("#filterBox").css('display', 'none');
		$("#loader").css('display', 'block');
		sessionStorage.setItem("currentQuery", query);
	}
}

function isAlphaNumeric(str) {
  var code, i, len;

  for (i = 0, len = str.length; i < len; i++) {
    code = str.charCodeAt(i);
    if (!(code > 47 && code < 58) && // numeric (0-9)
        !(code > 64 && code < 91) && // upper alpha (A-Z)
        !(code > 96 && code < 123) && // lower alpha (a-z)
        !([32,196,214,220,223,228,246,252].indexOf(code) >= 0  ) // space, sharp s, and german umlaut
        ) {
      return false;
    }
  }
  return true;
};

// refresh results
function refreshResults() {

	// make the status
	htmlContent = htmlStatus(response.metadata);

	// get filters
	var filters = $.grep($('.checkbox:checked').map(function() { return this.value; }).get(), function (n, i) {return n != 'all' && n != 'none' && n != 'invert'});

	filtered = response.sentences;
	filtered_urls = urls;

	// filter the results
	if (filters.length > 0){
		filtered = $.grep(filtered, function( n,i ) { return filters.includes(n.url.split('/')[2].replace('www.',''));})
		filtered_urls = $.grep(urls, function(n,i) { return filters.includes(n.split('/')[2].replace('www.',''));})
	}


	var numPages = Math.ceil(filtered.length / pageLength);


	// generate result as html
	switch(sessionStorage.view) {
		case 'Pro/Con':
			htmlContent += htmlProConView( filtered, false );
			numPages = Math.ceil(Math.max(
				$.grep(filtered, function( n, i ) { return n.stanceLabel=='pro'} ).length, $.grep(filtered, function( n, i ) { return n.stanceLabel=='contra'} ).length) / pageLength);
			break;
		case 'List':
			htmlContent += htmlListView( filtered );
			break;
		case 'Weights':
			htmlContent += htmlProConView( filtered, true );
			numPages = Math.ceil(Math.max(
				$.grep(filtered, function( n, i ) { return n.stanceLabel=='pro'} ).length, $.grep(filtered, function( n, i ) { return n.stanceLabel=='contra'} ).length) / pageLength);
			break;
		case 'Documents':
			htmlContent += htmlDocView( filtered, filtered_urls );
			numPages = Math.ceil(unique($.map(filtered, function( n, i ) { return n.url })).length / pageLength);
			break;
		default:
			break;
	}

	// generate paging
	if(filtered.length > 0){
		htmlContent += htmlPaging( numPages );
	}


	// set html of result container
	$(".resultBox").html(htmlContent);

	setPrevNext();

	$(".accordion").click(function() {
		$(this).toggleClass("showing");

		if ($(this).hasClass("showing")) {
			$(this).next().css('max-height', $(this).next().prop('scrollHeight') + 'px');
		} else {
			$(this).next().css('max-height', '');
		};
	});
}

function invertSelection() {
	$("[id^=checkbox]").each(function() {
    	this.checked = !this.checked;
	});

	page = 0;

	refreshResults();
}

function selectAll() {
	$("[id^=checkbox]").each(function() {
    	this.checked = true;
	});

	page = 0;

	refreshResults();
}

function selectNone() {
	$("[id^=checkbox]").each(function() {
    	this.checked = false;
	});

	page = 0;

	refreshResults();
}


function setPrevNext() {
	$("#prev").click(function(){
		page--;
		refreshResults();
	})

	$("#next").click(function(){
		page++;
		refreshResults();
	})

	$(".page").click(function(){
		page = parseInt($(this).text()) - 1;
		refreshResults();
	})
}

// generate the filter html code
function setupFilters() {

	// read the response
	obj = response.sentences

	var domains = $.map(obj, function( n, i ) { return n.url.split('/')[2].replace('www.','')});

	domainCounts = domains.reduce(function(res, e) {
		if(e in res){
			res[e] += 1;
		}else{
			res[e] = 1;
		}

		return res;
	}, {})

	domains = unique(domains);
	domains = domains.sort(function( a,b ) { return domainCounts[b] - domainCounts[a] })

	html = '';

	for( var i = 0; i < domains.length; i++ ){
		html += '<div class="filter-entry">\n'
        html += '<input type="checkbox" id="checkbox-' + i + '" class="checkbox" onclick="page=0;refreshResults();" value="' + domains[i] + '" />'
        html += '<label for="checkbox-' + i + '"></label><label for="checkbox-' + i + '" title="' + domains[i] + '";" class="checkbox-item">' + domains[i] + '</label>'
        html += '<label class="checkbox-item-amount">(' + domainCounts[domains[i]] + ')</label>\n</div>\n'
    }

	$("#filterSet").html( html );

	// add invert selection button

	// TODO: why is this done in JS??

	html = '<div class="filter-entry" style="float: right">\n'
    html += '<input type="checkbox" id="all" class="checkbox" onclick="selectAll();" value="all"/>'
    html += '<label for="all" title="select all"></label></div>'

    html += '<div class="filter-entry" style="float: right">\n'
    html += '<input type="checkbox" id="none" class="checkbox" onclick="selectNone();" value="none"/>'
    html += '<label for="none" title="select none"></label></div>'

    html += '<div class="filter-entry" style="float: right">\n'
    html += '<input type="checkbox" id="invert" class="checkbox" onclick="invertSelection();" value="invert"/>'
    html += '<label for="invert" title="invert selection"></label></div>'

	$("#controlSet").html( html );
}


function htmlDocView( sentences, urls ){
    html = '';

	var maxArgs = 0;

	for(var i = 0; i < urls.length; i++){
		if(i < urls.length){
			maxArgs = Math.max(maxArgs, $.grep(sentences, function( n, idx ) { return n.url==urls[i] && n.stanceLabel == 'pro'} ).length,
				$.grep(sentences, function( n, idx ) { return n.url==urls[i] && n.stanceLabel == 'contra'} ).length);
		}
	}

    for(var i = page * pageLength; i < (page + 1)* pageLength; i++){

        if(i < urls.length){

        	var args = $.grep(sentences, function( n, idx ) { return n.url==urls[i] } );

            var date = new Date(args[0].date)

            var domain = urls[i].split('/')[2].replace('www.','');

            var reg = new RegExp( '[?&]index=([^&#]*)', 'i' );
    		var string = reg.exec(window.location.href);
    		index = string!=null ? string[1] : 'cc';


            if ( index == 'nyt' ) {
                title = args[0].header;
            } else if ( docTitles==null || docTitles[i] == undefined) {
                title = domain;
            	//title = urls[i];
            } else {
            	title = docTitles[i];
            }

            numPro = $.grep(args, function( n, idx ) { return n.stanceLabel=='pro'} ).length
            numCon = $.grep(args, function( n, idx ) { return n.stanceLabel=='contra'} ).length

            pctPro = Math.ceil((numPro/maxArgs) * 100)
            pctCon = Math.ceil((numCon/maxArgs) * 100)

            entry = '<div class="accordion"><div class="histogram" title="Pro: ' + numPro + ', Con: ' + numCon + '"><div class="green" style="height:' + pctPro + '%"></div><div class="red" style="height:' + pctCon + '%"></div></div>\n'
            entry += '<div class="doctitle">' + title + '<br><a class="url" target="_blank" href="' + urls[i] + '">Source: ' + domain + '</a><span class="date">(' + date.toLocaleDateString('de-DE', {year: 'numeric', month: 'short', day: 'numeric'}) + ')</span></div></div>\n'
            entry += '<div class="panel">\n'

            counter = 0

            for(var j = 0; j < args.length; j++){
                arg = args[j]
                if(arg['url'] == urls[i]){
                    counter += 1
                    if(arg['stanceLabel'] == 'pro'){
                        entry += '<div class="result"><span class="pro">PRO:</span> '
                    }else{
                        entry += '<div class="result"><span class="con">CON:</span> '
                    }

                    conf = ((arg.argumentConfidence + arg.stanceConfidence) / 2).toFixed(4);
                    entry += '<span>'+ arg['sentenceOriginal'] + '</span> <span class="conf_score">('+ conf + ')</span></div>'
                }
            }

            entry += '</div>'

            html += entry + '\n'
        }
    }

    return html
}

function getDocList( sentences ) {

	// get list of urls
    urls = $.map(sentences, function( n, i ) { return n.url });

    counts = urls.reduce(function(res, e) {
		if(e in res){
			res[e] += 1;
		}else{
			res[e] = 1;
		}

		return res;
	}, {})

    urls = unique(urls);

	urls.sort(function( a,b ) { return counts[b] - counts[a] });

	var reg = new RegExp( '[?&]index=([^&#]*)', 'i' );
    var string = reg.exec(window.location.href);
    index = string!=null ? string[1] : 'cc';

    if (index=="cc" || index=="now" || index=="rss") {
		$.get( "/titles", { "urls": urls.join(","), "query": response.metadata.topic }, function( data ) {
			if (data.query == response.metadata.topic){
				docTitles = data.titles;
			}

 		});
    }
}

function unique(list) {
  var result = [];
  $.each(list, function(i, e) {
    if ($.inArray(e, result) == -1) result.push(e);
  });
  return result;
}

function replaceAll(str, find, replace) {
    return str.replace(new RegExp(escapeRegExp(find), 'g'), replace);
}

function escapeRegExp(str) {
    return str.replace(/([.*+?^=!:${}()|\[\]\/\\])/g, "\\$1");
}

function hideIntro() {
	$(".content").css('visibility', 'visible');
	$(".navBar").css('visibility', 'visible');
	$("#searchbar").removeClass('searchbarStart');
	$("#searchbar").addClass('searchbar');
	$("img").removeClass('imgStart');
	$("#header").removeClass('headerStart');
	$("#header").addClass('header');
	$("#intro").css('visibility', 'hidden');
}

function showIntro() {
	$(".content").css('visibility', 'hidden');
	$(".navBar").css('visibility', 'hidden');
	$("#searchbar").addClass('searchbarStart');
	$("#searchbar").removeClass('searchbar');
	$("img").addClass('imgStart');
	$("#header").addClass('headerStart');
	$("#header").removeClass('header');
	$("#queryInput").focus();
}

$(document).ready(function(){

	if ( !sessionStorage.queryResponse ){

		showIntro();

		$(".showLoader").click(hideIntro)
		$("#predictButton").click(hideIntro)

	} else {

		$("#intro").css('visibility', 'hidden');
		$("#loader").css('display', 'none');

		response = JSON.parse(sessionStorage.queryResponse);

		console.log(response)

		if (response["error"]){
			$(".resultBox").html('<span class="status">' + response['error'] + '</span>');
			$(".filterBox").css('visibility', 'hidden');
		} else {
			$("#queryInput").val(response.metadata.topic);
			getDocList(response.sentences);
			setupFilters();
			refreshResults();
		}

		if ( !sessionStorage.view ) {
			sessionStorage.view = 'Pro/Con';
		}

		$("#" + sessionStorage.view.replace('/', '')).addClass("active");

		//setPrevNext();

		$("#navBar > span").click(function() {
			$(this).addClass("active");
			$(this).siblings().removeClass("active");
			sessionStorage.view = $(this).text();
			page = 0;
			refreshResults();
		})

	}


});
