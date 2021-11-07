"use strict";

window.demo = {};


function plot(bifurcation_graph, graph) {
    bifurcation_graph.register_parameter('r')
	bifurcation_graph.register_function('func', "Math.sqrt(x) - r");

	graph.register_parameter('r');
	graph.register_function('func', document.getElementById('equation').value);
	graph.register_vector_field('vector_field', document.getElementById('equation').value);
	graph.register_fixed_points('fixed_points', document.getElementById('equation').value);

    bifurcation_graph.set_parameter('r', $('#slider').slider("option", "value"));
    graph.set_parameter('r', $('#slider').slider("option", "value"));
}

window.onload = function() {

      $("#parameter").text('0');
      var bifurcation_graph = new ds.graph.Graph("bifurcation_diagram");
      var graph = new ds.graph.Graph("graph");

      window.demo.plot = function() {
          plot(bifurcation_graph, graph);
      }


      $(".dropdown-menu").on("click", "li", function(event){
            $("#equation").val(event.currentTarget.innerText);
            plot(bifurcation_graph, graph);
      });

      $(function() {
        $( "#slider" ).slider({
            min: -10,
            max: 10,
            step: 0.001,
            slide: function(event, ui) {
                  $( "#parameter" ).text(ui.value);
                  graph.set_parameter('r', ui.value);
		          bifurcation_graph.set_parameter('r', ui.value);
            }
        });
    });
}


