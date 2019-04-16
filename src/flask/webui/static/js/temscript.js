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
