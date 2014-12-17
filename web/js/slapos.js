(function(angular) {
    angular.module('slapOS', []).controller('SlapOSController', function($scope) {
        $scope.penis = "balls";
        $scope.change = function(){
            console.log($scope.log_message)
        };
    });
})(window.angular);