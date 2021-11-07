"use strict";

 // http://tutorials.jenkov.com/svg/marker-element.html

window.ds = {version : "1.0"};


(function create_numerical_package(ds) {
    ds.numeric = {};

    ds.numeric.is_stable = function(fn, x) {
        var h = 0.01;
        return Math.sign(fn(x + h) - fn(x)) == -1;
    };
    
    ds.numeric.get_absolute_error = function(x, y) {
    	return Math.abs(x-y);
    };

    ds.numeric.arange = function(x_lower, x_upper, step) {
	/*
	 * Mimics the functionality from numpy.arange
	 */
        var x_array = [];
	var x = x_lower;

	if (x_upper < x_lower) {
	    throw "x_upper cannot be lower than x_lower";
	}

	if (step <= 0.0) {
	    throw "step cannot be less than or equal to 0";
	}
	
	for(var i = 0; x < x_upper; i++) {
	    x = x_lower + i*step;
	    x_array.push(x);
	}

	return x_array;
    };

    ds.numeric.get_roots = function(fn, x_lower, x_upper, algorithm) {
	var roots;
	if (algorithm.toLowerCase() === "bisection") {
	    roots = get_roots_using_bisection_algorithm(fn, x_lower, x_upper);
	} else {
	    throw "algorithm not supported";
	}

	return roots;
    };

    var get_roots_using_bisection_algorithm = function (fn, x_lower, x_upper) {
        var get_root = function(fn, x_lower, x_upper, x_root) {
	    
	    var eps = 0.01;
            var x_mid = (x_lower + x_upper) / 2.0;
            var r;

            if ( fn(x_lower) * fn(x_mid) <= 0.0 ) {
                r = (x_lower + x_mid) / 2.0;
                if (x_root === undefined) {
                    x_root = r;
                } else if (ds.numeric.get_absolute_error(x_root, r) < eps) {
                    return r;
                }

                return get_root(fn, x_lower, x_mid, r);
            } else if ( fn(x_mid) * fn(x_upper) <= 0.0 ) {
                r = (x_mid + x_upper) / 2.0;
                if (x_root === undefined) {
                    x_root = r;
                } else if (ds.numeric.get_absolute_error(x_root, r) < eps) {
                    return r;
		}

                return get_root(fn, x_mid, x_upper, x_root);
            }
            return r;
        };

        var roots = [];
	var step = 0.1;
	var x = ds.numeric.arange(x_lower, x_upper, step);
	for(var i = 1; i < x.length; i++) {
	    if (fn(x[i-1]) * fn(x[i]) <= 0.0) {
                roots.push(get_root(fn, x[i-1], x[i]));
		i += 1;
	    }
	} // end for

	return roots;
    }	    
})(window.ds);


(function create_graph_package(ds) {

    ds.graph = {};

    /* Helper functions */
    function getLineForPath(x1, y1, x2, y2) {
            return "M" + x1 + "," + y1 + " L" + x2 +"," + y2 +"";
    }

    ds.graph.Parameter = Parameter
    function Parameter(name) {
        this._name = name;
        this._value;
        this._observers = [];
    }

    Parameter.prototype.add_observer = function(o) {
        this._observers.push(o);
    }

    Parameter.prototype.set_value = function(value) {
        this._value = value;
        this._notify_observers();
    }

    Parameter.prototype._notify_observers = function() {
        for(var i = 0 ; i < this._observers.length; i++) {
            this._observers[i].set_parameter(this._name, this._value);
            this._observers[i].visualize();
        }
    }

    ds.graph.VectorField = VectorField;
    function VectorField(name, value, g) {
        this._name = name;
        this._value = value;
        this._xlim = [-4.0, 4.0];
        this._h = 0.5;
        this._parameters = {};
        this._graph = g;
    }



    VectorField.prototype.visualize = function() {

        var selection = this._graph._svg.selectAll("#" + this._name + " path");
        var x,y;
        var data = [];
        var t = 0.1;
        var delta;

        for(var p in this._parameters) {
            // TODO: Find better way to do this. Don't attach to global scope
            window[p] = this._parameters[p];
        }

        for (var x = this._xlim[0]; x < this._xlim[1]; x += this._h) {
            y = eval(this._value);

            if (y < 0) {
                delta = -t;
            } else {
                delta = t;
            }

            data.push({'x1' : x, 'y1' : 0, 'x2' : x + delta, 'y2' : 0});
        }

        if (selection.empty()) {
            this._graph.plot_vectors(this._name, data, 'black');
        } else {
            selection.data(data)
                     .attr("d",(function(d) {return getLineForPath( this._graph._xScale(d.x1),  this._graph._yScale(d.y1),  this._graph._xScale(d.x2),  this._graph._yScale(d.y2)) }).bind(this));

        }
    }

    VectorField.prototype.zoom = function() {

        var s = d3.event.scale;
        var pixel_width = 700; // A constant
        var num_width = 8.0/s;

        var x_shift = d3.event.translate[0] * (num_width/pixel_width);
        var b = num_width/2.0;

        var l = -(b + x_shift);
        var r = b - x_shift;

        this._xlim = [l,r];
        this._h = 0.5/s;


        var selection = this._graph._svg.selectAll("#" + this._name + " path");
        var x,y;
        var data = [];

        for(var p in this._parameters) {
            // TODO: Find better way to do this. Don't attach to global scope
            window[p] = this._parameters[p];
        }

        for (var x = this._xlim[0]; x < this._xlim[1]; x += this._h) {
            y = eval(this._value);
            data.push({'x1' : x, 'y1' : 0, 'x2' : x +  (Math.sign(y)*0.1)/s, 'y2' : 0});
        }

        selection.data(data)
                     .attr("d",(function(d) {return getLineForPath( this._graph._xScale(d.x1),  this._graph._yScale(d.y1),  this._graph._xScale(d.x2),  this._graph._yScale(d.y2)) }).bind(this));
    }

    VectorField.prototype.set_parameter = function(name, value) {
        this._parameters[name] = value;
    }

   


    ds.graph.FixedPoints = FixedPoints;
    function FixedPoints(name, value, g) {
        this._name = name;
	    this._value = value;
	    this._xlim = [-4.0, 4.0];
	    this._parameters = {};
	    this._graph = g;
    }

    FixedPoints.prototype.visualize = function() {
        var selection = this._graph._svg.selectAll("#" + this._name + " circle");
	    var roots;
	    var data = [];
	
	    for(var p in this._parameters) {
	        // TODO: Find better way to do this. Don't attach to global scope
	        window[p] = this._parameters[p];
	    };

	    var fn = function(x) {
	        return eval(this._value);
	    };

	    roots = ds.numeric.get_roots(fn.bind(this), this._xlim[0], this._xlim[1], "bisection");

	    for (var i = 0; i < roots.length; i++) {
	        data.push({'x' : roots[i], 'y' : 0.0, 'stable' : ds.numeric.is_stable(fn.bind(this), roots[i])});
	    };

	    if (data.length > 0) {
	        // Data is not empty
	        if (selection.empty()) {
	            this._graph.plot_fixed_points(this._name, data, 10.0, "yellow");
	        } else {
	            selection.data(data)
		                  .attr("cx", (function(d) {return this._graph._xScale(d.x)}).bind(this))
                          .attr("cy", (function(d) {return this._graph._yScale(d.y)}).bind(this));
	        }
	    } else {
	        // Remove fixed points from graph
		    selection.remove();
	    }
    }

    FixedPoints.prototype.zoom = function() {
        var selection = this._graph._svg.selectAll("#" + this._name + " circle");
        selection.attr("cx", (function(d) {return this._graph._xScale(d.x)}).bind(this))
        selection.attr("cy", (function(d) {return this._graph._yScale(d.y)}).bind(this));
    }

    FixedPoints.prototype.set_parameter = function(name, value) {
        this._parameters[name] = value;
    }


    ds.graph.Function = Function;
    function Function(name, value, g) {
        this._name = name;
        this._value = value;
        this._xlim = [-4.0, 4.0];
        this._h = 0.01;
        this._parameters = {};
        this._graph = g;
    }

    Function.prototype.visualize = function() {

        var selection = this._graph._svg.selectAll("#" + this._name + " circle");
        var x,y;
        var data = [];

        for(var p in this._parameters) {
            // TODO: Find better way to do this. Don't attach to global scope
            window[p] = this._parameters[p];
        }

        for (var x = this._xlim[0]; x < this._xlim[1]; x += this._h) {
            y = eval(this._value);
            data.push({'x' : x, 'y' : y});
        }

        if (selection.empty()) {
            this._graph.plot_points(this._name, data, 1.0,'yellow');
        } else {
            selection.data(data)
                     .attr("cx", (function(d) {return this._graph._xScale(d.x)}).bind(this))
                     .attr("cy", (function(d) {return this._graph._yScale(d.y)}).bind(this));

        }
    }

    Function.prototype.zoom = function() {
        var selection = this._graph._svg.selectAll("#" + this._name + " circle");
        selection.attr("cx", (function(d) {return this._graph._xScale(d.x)}).bind(this))
        selection.attr("cy", (function(d) {return this._graph._yScale(d.y)}).bind(this));
    }

    Function.prototype.set_parameter = function(name, value) {
        this._parameters[name] = value;
    }

    ds.graph.Graph = Graph;

    function Graph(element_id) {
       this._svg;
       this._functions = {};
       this._vector_fields = {};
       this._fixed_points = {};
       this._parameters = {};
       this._svg_dim = {'width' : 350, 'height' : 350};
       this._xScale;
       this._yScale;
       this._xAxis;
       this._yAxis;
       this._init(element_id);
    }

    Graph.prototype.init_scales = function(x1, x2, y1, y2) {
          this._xScale = d3.scale.linear();
          this._yScale = d3.scale.linear();     

          this._xScale.domain([x1, x2])
                      .range([0, this._svg_dim.width]);
          

          this._yScale.domain([y1, y2])
                      .range([this._svg_dim.height, 0]);

    }

    Graph.prototype.init_axes = function() {

        // Invoke the function and store it.
        this._xAxis = d3.svg.axis();
        this._yAxis = d3.svg.axis();

        this._xAxis.orient("bottom")
                   .ticks(10)
                   .scale(this._xScale);

        this._yAxis.orient("left")
                   .ticks(10)
                   .scale(this._yScale);


         var group = this._svg.append("g");


         group.append("g")
              .attr("class", "y-axis axis")
              .attr("transform", "translate(" + this._xScale(0.0) + ",0)");

         group.append("g")
              .attr("class", "x-axis axis")
              .attr("transform", "translate(0,"+ this._yScale(0.0) +")");

         // Update visualization
         this._svg.select(".x-axis").call(this._xAxis);
         this._svg.select(".y-axis").call(this._yAxis);
    }


    Graph.prototype.init_zoom = function() {
          var zoom = d3.behavior.zoom();

          zoom.x(this._xScale);
          zoom.y(this._yScale);

          // Used to decide on how to handle zoom and panning gestures
          zoom.on("zoom", (function(){

              this._svg.select(".x-axis").call(this._xAxis);
              this._svg.select(".y-axis").call(this._yAxis);

              var x = this._xScale(0.0);
              var margin = 20;

              if (this._svg_dim['width'] - margin > x && x > margin) {
                  this._svg.select(".y-axis").attr("transform", "translate(" + x + ",0)");
              }

              var y = this._yScale(0.0);
              if (this._svg_dim['height'] - margin > y && y > margin) {
                  this._svg.select(".x-axis").attr("transform", "translate(0," +  y + ")");
              }

              for (var k in this._functions) {
                this._functions[k].zoom();
              }

              for (var k in this._vector_fields) {
                this._vector_fields[k].zoom(d3.scale);
              }

              for (var k in this._fixed_points) {
                this._fixed_points[k].zoom(d3.scale);
              }
         }).bind(this));

         // We use the svg selection and instruct it to listen for zooming and panning gestures
         this._svg.call(zoom);
    }

     Graph.prototype.init_markers = function() {

         var defs = this._svg.append("defs");

         defs.append("marker")
             .attr("id", "markerCircle")
             .attr("markerWidth", 1)
             .attr("markerHeight", 1)
             .attr("refX", 5)
             .attr("refY", 5)
             .append("circle")
             .attr("stroke", "black")
             .attr("cx", 5)
             .attr("cy", 5)
             .attr("r", 3)

         defs.append("marker")
             .attr("id", "markerArrow")
             .attr("markerWidth", 10)
             .attr("markerHeight", 10)
             .attr("refX", 2)
             .attr("refY", 6)
             .attr("orient", "auto")
             .append("path")
             .attr("d", "M2,2 L2,11 L10,6 L2,2")
     }

    Graph.prototype._init = function(element_id) {
          this._svg = d3.select("#" + element_id)
                        .append("svg")
                        .attr("width",  this._svg_dim.width)
                        .attr("height", this._svg_dim.height)

          this.init_scales(-4, 4, -4, 4);
          this.init_axes();
          this.init_zoom();
          this.init_markers();
    }

    Graph.prototype.register_function = function(name, value) {
        var fn = new Function(name, value, this);

        // Register function to listen for parameter change if applicable
        for (var k in this._parameters) {
            if (value.indexOf(k) !== -1) {
                this._parameters[k].add_observer(fn);
            }
        }

        this._functions[name] = fn;
    }

    Graph.prototype.register_fixed_points = function(name, value) {
	    var fn = new FixedPoints(name, value, this);
	    
	    // Register to lister fo parameter change
	    for (var k in this._parameters) {
	        if(value.indexOf(k) !== -1) {
	            this._parameters[k].add_observer(fn);
	        }
	    }

	    this._fixed_points[name] = fn;
    }

    Graph.prototype.register_vector_field = function(name, value) {
        var vf = new VectorField(name, value, this);

        // Register function to listen for parameter change if applicable
        for (var k in this._parameters) {
            if (value.indexOf(k) !== -1) {
                this._parameters[k].add_observer(vf);
            }
        }

        this._vector_fields[name] = vf;
    }


    Graph.prototype.register_parameter = function(name) {
        this._parameters[name] = new Parameter(name);
    }

     Graph.prototype.set_parameter = function(name, value) {
        this._parameters[name].set_value(value);
    }

    Graph.prototype.plot_points = function(name, data, r, color) {
          this._svg.append("g")
                   .attr("id", name)
                   .selectAll("circle")
                   .data(data)
                   .enter()
                   .append("circle")
                   .attr("stroke", color)
                   .attr("fill", color)
                   .attr("cx", (function(d) {return this._xScale(d.x)}).bind(this))
                   .attr("cy", (function(d) {return this._yScale(d.y)}).bind(this))
                   .attr("r", r);
    }

    Graph.prototype.plot_fixed_points = function(name, data, r, color) {
          this._svg.append("g")
                   .attr("id", name)
                   .selectAll("circle")
                   .data(data)
                   .enter()
                   .append("circle")
                   .attr("stroke", color)
                   .attr("fill", (function(d) {if (d.stable) return color; else return "none";}))
                   .attr("cx", (function(d) {return this._xScale(d.x)}).bind(this))
                   .attr("cy", (function(d) {return this._yScale(d.y)}).bind(this))
                   .attr("r", r);
    }

    Graph.prototype.plot_vectors = function(name, data, color) {

         this._svg.append("g")
                   .attr("id", name)
                   .selectAll("path")
                   .data(data)
                   .enter()
                   .append("path")
                   .attr("stroke", color)
                   .attr("stroke-width", 2)
                   .attr("marker-end", "url(#markerArrow)")
                   .attr("d",(function(d) {return getLineForPath( this._xScale(d.x1),  this._yScale(d.y1),  this._xScale(d.x2),  this._yScale(d.y2)) }).bind(this));
    }
})(window.ds);

