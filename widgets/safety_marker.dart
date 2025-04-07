import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';

enum MarkerType {
  police,
  safe,
  danger,
}

class SafetyMarker {
  static BitmapDescriptor getMarkerIcon(MarkerType type) {
    switch (type) {
      case MarkerType.police:
        return BitmapDescriptor.defaultMarkerWithHue(BitmapDescriptor.hueBlue);
      case MarkerType.safe:
        return BitmapDescriptor.defaultMarkerWithHue(BitmapDescriptor.hueGreen);
      case MarkerType.danger:
        return BitmapDescriptor.defaultMarkerWithHue(BitmapDescriptor.hueRed);
    }
  }

  static String getMarkerTitle(MarkerType type) {
    switch (type) {
      case MarkerType.police:
        return 'Police Station';
      case MarkerType.safe:
        return 'Safe Zone';
      case MarkerType.danger:
        return 'Danger Zone';
    }
  }

  static Color getMarkerColor(MarkerType type) {
    switch (type) {
      case MarkerType.police:
        return Colors.blue;
      case MarkerType.safe:
        return Colors.green;
      case MarkerType.danger:
        return Colors.red;
    }
  }
} 