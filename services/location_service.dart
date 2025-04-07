import 'package:geolocator/geolocator.dart';
import 'package:permission_handler/permission_handler.dart';

class LocationData {
  final double latitude;
  final double longitude;
  final String? address;

  LocationData({
    required this.latitude,
    required this.longitude,
    this.address,
  });
}

class LocationService {
  Future<bool> _handleLocationPermission() async {
    bool serviceEnabled;
    LocationPermission permission;

    serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      return false;
    }

    permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        return false;
      }
    }

    if (permission == LocationPermission.deniedForever) {
      return false;
    }

    return true;
  }

  Future<LocationData> getCurrentLocation() async {
    final hasPermission = await _handleLocationPermission();
    if (!hasPermission) {
      throw Exception('Location permissions are denied');
    }

    try {
      final position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );

      return LocationData(
        latitude: position.latitude,
        longitude: position.longitude,
      );
    } catch (e) {
      throw Exception('Error getting location: $e');
    }
  }

  Future<List<LocationData>> getNearbyPoliceStations(LocationData currentLocation) async {
    // Implement API call to get nearby police stations
    // This is a placeholder for the actual implementation
    return [];
  }

  Future<List<LocationData>> getSafeZones(LocationData currentLocation) async {
    // Implement API call to get safe zones
    // This is a placeholder for the actual implementation
    return [];
  }

  Future<List<LocationData>> getDangerZones(LocationData currentLocation) async {
    // Implement API call to get danger zones
    // This is a placeholder for the actual implementation
    return [];
  }
} 