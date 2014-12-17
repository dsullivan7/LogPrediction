(function(angular, slapos_model) {
    angular.module('slapOS', []).controller('SlapOSController', function($scope) {
        $scope.weights = slapos_model["weights"]
        $scope.intercept = slapos_model["intercept"]
        $scope.mean = slapos_model["scale"]["mean"]
        $scope.std = slapos_model["scale"]["std"]

        $scope.change = function(){
            $scope.result = ""
            var entries = $scope.log_message.split(",")
            var entries = [entries[3], entries[5], entries[6], entries[7]]

            var dot = 0.0
            for(var i in entries){
                entries[i] -= $scope.mean[i]
                entries[i] /= $scope.std[i]
                dot += entries[i] * $scope.weights[i]
            }

            dot += $scope.intercept
            dot *= -1.0
            dot = Math.exp(dot)
            dot += 1.0
            dot = 1.0 / dot

            var pre_message = "The probabilty that this user will cause a problem is "
            if(!isNaN(parseFloat(dot)) && isFinite(dot)){
                $scope.result = pre_message + dot
            }
        };
    });
})(window.angular, slapos_model);