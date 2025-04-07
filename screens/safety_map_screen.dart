import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import '../services/location_service.dart';
import '../widgets/safety_marker.dart';

class SafetyMapScreen extends StatefulWidget {
  const SafetyMapScreen({super.key});

  @override
  State<SafetyMapScreen> createState() => _SafetyMapScreenState();
}

class _SafetyMapScreenState extends State<SafetyMapScreen> {
  final LocationService _locationService = LocationService();
  GoogleMapController? _mapController;
  Set<Marker> _markers = {};
  LocationData? _currentLocation;

  @override
  void initState() {
    super.initState();
    _initializeMap();
  }

  Future<void> _initializeMap() async {
    try {
      final location = await _locationService.getCurrentLocation();
      setState(() {
        _currentLocation = location;
        _updateMarkers();
      });
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error getting location: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _updateMarkers() async {
    if (_currentLocation == null) return;

    final policeStations = await _locationService.getNearbyPoliceStations(_currentLocation!);
    final safeZones = await _locationService.getSafeZones(_currentLocation!);
    final dangerZones = await _locationService.getDangerZones(_currentLocation!);

    setState(() {
      _markers = {
        ..._createMarkers(policeStations, MarkerType.police),
        ..._createMarkers(safeZones, MarkerType.safe),
        ..._createMarkers(dangerZones, MarkerType.danger),
      };
    });
  }

  Set<Marker> _createMarkers(List<LocationData> locations, MarkerType type) {
    return locations.map((location) {
      return Marker(
        markerId: MarkerId('${type.toString()}_${location.latitude}_${location.longitude}'),
        position: LatLng(location.latitude, location.longitude),
        icon: SafetyMarker.getMarkerIcon(type),
      );
    }).toSet();
  }

  @override
  Widget build(BuildContext context) {
    if (_currentLocation == null) {
      return const Scaffold(
        body: Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Safety Map'),
        centerTitle: true,
      ),
      body: GoogleMap(
        initialCameraPosition: CameraPosition(
          target: LatLng(_currentLocation!.latitude, _currentLocation!.longitude),
          zoom: 15,
        ),
        markers: _markers,
        myLocationEnabled: true,
        myLocationButtonEnabled: true,
        onMapCreated: (controller) {
          _mapController = controller;
        },
      ),
    );
  }

  @override
  void dispose() {
    _mapController?.dispose();
    super.dispose();
  }
} 